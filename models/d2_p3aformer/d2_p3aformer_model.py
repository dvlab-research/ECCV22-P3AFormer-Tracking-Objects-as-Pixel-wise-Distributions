from turtle import onkey
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

# from .mask2former_modeling.criterion import SetCriterion
# from .mask2former_modeling.matcher import HungarianMatcher
from .transcenter_losses.utils import _sigmoid
from .transcenter_losses.losses import FastFocalLoss, RegWeightedL1Loss, loss_boxes
from util.p3aformer.p3aformer_misc import NestedTensor
from .transcenter_post_processing.decode import generic_decode
from .transcenter_post_processing.post_process import generic_post_process
from .transcenter_backbone import build_backbone
from .p3aformer_deformable_transformer import build_deforamble_transformer
import copy
import util.p3aformer.p3aformer_misc as p3aformer_utils
from .transcenter_dla import IDAUpV3
from torch import Tensor
import pdb


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class GenericLoss(torch.nn.Module):
    def __init__(self, cfg, weight_dict):
        super(GenericLoss, self).__init__()
        self.crit = FastFocalLoss()
        self.crit_reg = RegWeightedL1Loss()
        self.weight_dict = weight_dict
        self.allow_missing = True
        self.dec_layers = cfg.MODEL.DENSETRACK.DEC_LAYERS
        self.norm_factor = cfg.MODEL.DENSETRACK.NORM_FACTOR

    def _sigmoid_output(self, output):
        if "hm" in output:
            output["hm"] = _sigmoid(output["hm"])
        return output

    def forward(self, outputs, batch):
        regression_heads = ["reg", "wh", "tracking", "center_offset"]
        losses = {}
        outputs = self._sigmoid_output(outputs)
        for s in range(self.dec_layers):
            if s < self.dec_layers - 1:
                end_str = f"_{s}"
            else:
                end_str = ""
            # only 'hm' is using focal loss for heatmap regression. #
            if "hm" in outputs:
                losses["hm" + end_str] = (
                    self.crit(
                        outputs["hm"][s],
                        batch["hm"],
                        batch["ind"],
                        batch["mask"],
                        batch["cat"],
                    )
                    / self.norm_factor
                )

            for head in regression_heads:
                if head in outputs:
                    if self.allow_missing and (
                        head + "_mask" not in batch or head not in outputs
                    ):
                        continue
                    if outputs[head] is None or len(outputs[head]) == 0:
                        continue
                    losses[head + end_str] = (
                        self.crit_reg(
                            outputs[head][s],
                            batch[head + "_mask"],
                            batch["ind"],
                            batch[head],
                        )
                        / self.norm_factor
                    )
            losses["boxes" + end_str], losses["giou" + end_str] = loss_boxes(
                outputs["boxes"][s], batch
            )
            losses["boxes" + end_str] /= self.norm_factor
            losses["giou" + end_str] /= self.norm_factor
        return losses


@META_ARCH_REGISTRY.register()
class D2P3AFormer(nn.Module):
    """
    Main class for p3aformer modeling.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone,
        transformer,
        num_classes,
        num_feature_levels,
        aux_loss=True,
        eval=True,
        criterion=None,
    ):
        super().__init__()
        self.transformer = transformer
        self.eval_mode = eval
        self.dense_fusion = True
        self.use_prev = True
        hidden_dim = transformer.d_model

        if self.dense_fusion:
            self.ida_up = IDAUpV3(64, [256, 256, 256, 256], [2, 4, 8, 16])
            self.linear_compress = nn.Linear(512, 256)
        else:
            self.ida_up = IDAUpV3(64, [256, 256, 256, 256], [2, 4, 8, 16])
            self.linear_compress = None

        self.ida_up = _get_clones(self.ida_up, 2)
        self.hm = nn.Sequential(
            nn.Conv2d(
                64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.ct_offset = nn.Sequential(
            nn.Conv2d(
                64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.reg = nn.Sequential(
            nn.Conv2d(
                64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        self.wh = nn.Sequential(
            nn.Conv2d(
                64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        # future tracking offset
        self.tracking = (
            nn.Sequential(
                nn.Conv2d(
                    129,
                    256,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=3 // 2,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True),
            )
            if self.use_prev
            else None
        )

        # init weights #
        # prior bias
        self.hm[-1].bias.data.fill_(-4.6)
        fill_fc_weights(self.reg)
        fill_fc_weights(self.wh)
        fill_fc_weights(self.ct_offset)
        if self.use_prev:
            fill_fc_weights(self.tracking)

        self.num_feature_levels = num_feature_levels

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone
        self.aux_loss = aux_loss

        # init weights #
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        self.transformer.decoder.reg = None
        self.transformer.decoder.ida_up = None
        self.transformer.decoder.wh = None
        self.query_embed = None
        self.criterion = criterion

    @property
    def device(self):
        return self.hm[-1].bias.data.device

    @classmethod
    def from_config(cls, cfg):
        num_classes = 1 if cfg.INPUT.DATASET_MAPPER_NAME != "coco" else 80
        backbone = build_backbone(cfg)
        transformer = build_deforamble_transformer(cfg)
        weight_dict = {
            "hm": cfg.MODEL.DENSETRACK.HM_WEIGHT,
            "reg": cfg.MODEL.DENSETRACK.OFF_WEIGHT,
            "wh": cfg.MODEL.DENSETRACK.WH_WEIGHT,
            "boxes": cfg.MODEL.DENSETRACK.BOXES_WEIGHT,
            "giou": cfg.MODEL.DENSETRACK.GIOU_WEIGHT,
            "center_offset": cfg.MODEL.DENSETRACK.CT_OFFSET_WEIGHT,
            "tracking": cfg.MODEL.DENSETRACK.TRACKING_WEIGHT,
        }
        if cfg.SOLVER.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(cfg.MODEL.DENSETRACK.DEC_LAYERS - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        criterion = (
            None
            if cfg.SOLVER.EVAL
            else GenericLoss(cfg, weight_dict).to(cfg.MODEL.DEVICE)
        )
        return {
            "backbone": backbone,
            "transformer": transformer,
            "num_classes": num_classes,
            "num_feature_levels": cfg.MODEL.DENSETRACK.NUM_FEATURE_LEVELS,
            "aux_loss": cfg.SOLVER.AUX_LOSS,
            "eval": cfg.SOLVER.EVAL,
            "criterion": criterion,
        }

    def collate_fn(self, data_list):
        data_dict = {}
        for one_k in data_list[0]:
            data_dict[one_k] = torch.stack(
                [torch.tensor(ele[one_k]) for ele in data_list]
            ).to(self.device)
        targets = {
            k: v
            for k, v in data_dict.items()
            if k != "orig_image"
            and k != "image"
            and "pad_mask" not in k
            and "pre_img" not in k
        }
        hm = data_dict["hm"]
        # pre_hm = data_dict["pre_hm"]
        samples = p3aformer_utils.NestedTensor(
            data_dict["image"], data_dict["pad_mask"]
        )
        pre_samples = p3aformer_utils.NestedTensor(
            data_dict["pre_img"], data_dict["pre_pad_mask"]
        )
        features = None
        pos = None
        pre_features = None
        pre_pos = None
        return (
            samples,
            pre_samples,
            None,
            hm,
            features,
            pos,
            pre_features,
            pre_pos,
            targets,
        )

    def get_detection_output(
        self, samples, pre_samples, pre_hm, features, pos, pre_features, pre_pos
    ):
        assert isinstance(samples, NestedTensor)
        if self.use_prev:
            assert isinstance(pre_samples, NestedTensor)
        if features is None:
            features, pos = self.backbone(samples)
        srcs = []
        masks = []
        outputs_coords = []
        outputs_hms = []
        outputs_regs = []
        outputs_whs = []
        outputs_ct_offsets = []
        outputs_tracking = []

        if self.use_prev:
            with torch.no_grad():
                if (pre_features is None and self.eval_mode) or not self.eval_mode:
                    pre_features, pre_pos = self.backbone(pre_samples)
                pre_srcs = []
                pre_masks = []

            for l, (feat, pre_feat) in enumerate(zip(features, pre_features)):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)

                pre_src, pre_mask = pre_feat.decompose()
                pre_srcs.append(self.input_proj[l](pre_src))
                pre_masks.append(pre_mask)

                assert mask is not None
                assert pre_mask is not None
                assert pre_src.shape == src.shape

            # make mask, src, pos embed
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                        pre_src = self.input_proj[l](pre_features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                        pre_src = self.input_proj[l](pre_srcs[-1])
                    assert pre_src.shape == src.shape
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                        torch.bool
                    )[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)

                    # pre
                    pre_m = pre_samples.mask
                    pre_mask = F.interpolate(
                        pre_m[None].float(), size=pre_src.shape[-2:]
                    ).to(torch.bool)[0]
                    pre_pos_l = self.backbone[1](NestedTensor(pre_src, pre_mask)).to(
                        src.dtype
                    )
                    pre_srcs.append(pre_src)
                    pre_masks.append(pre_mask)
                    pre_pos.append(pre_pos_l)

            if self.query_embed is not None:
                query_embed = self.query_embed.weight
            else:
                query_embed = None
            merged_hs = self.transformer(
                srcs,
                masks,
                pos,
                query_embed,
                pre_srcs=pre_srcs,
                pre_masks=pre_masks,
                pre_hms=None,
                pre_pos_embeds=pre_pos,
            )
            if self.dense_fusion:
                hs = [ele[0] for ele in merged_hs]
                for layer_lvl in range(len(hs)):
                    if self.dense_fusion:
                        hs[layer_lvl] = [ele[:, :256, :, :] for ele in hs[layer_lvl]]
                    hs[layer_lvl] = self.ida_up[0](
                        hs[layer_lvl], 0, len(hs[layer_lvl])
                    )[-1]
                    ct_offset = self.ct_offset(hs[layer_lvl])
                    wh_head = self.wh(hs[layer_lvl])
                    reg_head = self.reg(hs[layer_lvl])
                    hm_head = self.hm(hs[layer_lvl])
                    outputs_whs.append(wh_head)
                    outputs_ct_offsets.append(ct_offset)
                    outputs_regs.append(reg_head)
                    outputs_hms.append(hm_head)

                    # b,2,h,w => b,4,h,w
                    if not self.eval_mode:
                        outputs_coords.append(
                            torch.cat([reg_head + ct_offset, wh_head], dim=1)
                        )
                out = {
                    "hm": torch.stack(outputs_hms),
                    "wh": torch.stack(outputs_whs),
                    "reg": torch.stack(outputs_regs),
                    "center_offset": torch.stack(outputs_ct_offsets),
                }
                if not self.eval_mode:
                    out["boxes"] = torch.stack(outputs_coords)
            else:
                hs = []
                pre_hs = []
                pre_hm_out = F.interpolate(
                    pre_hm.float(), size=(pre_hm.shape[2] // 4, pre_hm.shape[3] // 4)
                )
                for hs_m, pre_hs_m in merged_hs:
                    hs.append(hs_m)
                    pre_hs.append(pre_hs_m)
                for layer_lvl in range(len(hs)):
                    hs[layer_lvl] = self.ida_up[0](
                        hs[layer_lvl], 0, len(hs[layer_lvl])
                    )[-1]
                    pre_hs[layer_lvl] = self.ida_up[1](
                        pre_hs[layer_lvl], 0, len(pre_hs[layer_lvl])
                    )[-1]

                    ct_offset = self.ct_offset(hs[layer_lvl])
                    wh_head = self.wh(hs[layer_lvl])
                    reg_head = self.reg(hs[layer_lvl])
                    hm_head = self.hm(hs[layer_lvl])

                    tracking_head = self.tracking(
                        torch.cat(
                            [hs[layer_lvl].detach(), pre_hs[layer_lvl], pre_hm_out],
                            dim=1,
                        )
                    )

                    outputs_whs.append(wh_head)
                    outputs_ct_offsets.append(ct_offset)
                    outputs_regs.append(reg_head)
                    outputs_hms.append(hm_head)
                    outputs_tracking.append(tracking_head)

                    # b,2,h,w => b,4,h,w
                    if not self.eval_mode:
                        outputs_coords.append(
                            torch.cat([reg_head + ct_offset, wh_head], dim=1)
                        )
                out = {
                    "hm": torch.stack(outputs_hms),
                    "wh": torch.stack(outputs_whs),
                    "reg": torch.stack(outputs_regs),
                    "center_offset": torch.stack(outputs_ct_offsets),
                    "tracking": torch.stack(outputs_tracking),
                }
                if not self.eval_mode:
                    out["boxes"] = torch.stack(outputs_coords)
        else:
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)

            # make mask, src, pos embed
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                        torch.bool
                    )[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)

            if self.query_embed is not None:
                query_embed = self.query_embed.weight
            else:
                query_embed = None
            merged_hs = self.transformer(
                srcs,
                masks,
                pos,
                query_embed,
                pre_srcs=None,
                pre_masks=None,
                pre_hms=None,
                pre_pos_embeds=None,
            )
            hs = [m[0] for m in merged_hs]
            for layer_lvl in range(len(hs)):
                hs[layer_lvl] = self.ida_up[0](hs[layer_lvl], 0, len(hs[layer_lvl]))[-1]
                ct_offset = self.ct_offset(hs[layer_lvl])
                wh_head = self.wh(hs[layer_lvl])
                reg_head = self.reg(hs[layer_lvl])
                hm_head = self.hm(hs[layer_lvl])

                outputs_whs.append(wh_head)
                outputs_ct_offsets.append(ct_offset)
                outputs_regs.append(reg_head)
                outputs_hms.append(hm_head)

                # b,2,h,w => b,4,h,w
                if not self.eval_mode:
                    outputs_coords.append(
                        torch.cat([reg_head + ct_offset, wh_head], dim=1)
                    )
            out = {
                "hm": torch.stack(outputs_hms),
                "wh": torch.stack(outputs_whs),
                "reg": torch.stack(outputs_regs),
            }
            if not self.eval_mode:
                out["boxes"] = torch.stack(outputs_coords)
        return out

    def forward(self, data_dict):
        # convert to torch and send to device
        (
            samples,
            pre_samples,
            pre_hm,
            hm,
            features,
            pos,
            pre_features,
            pre_pos,
            targets,
        ) = self.collate_fn(data_dict)
        out = self.get_detection_output(
            samples, pre_samples, pre_hm, features, pos, pre_features, pre_pos
        )
        loss_dict = self.criterion(out, targets)
        loss_dict_weighted = {
            k: v * self.criterion.weight_dict[k]
            for k, v in loss_dict.items()
            if k in self.criterion.weight_dict
        }
        return loss_dict_weighted


def build(args):
    num_classes = 1 if args.dataset_file != "coco" else 80

    if args.dataset_file == "coco":
        valid_ids = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            27,
            28,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            67,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
        ]
    else:
        valid_ids = [1]

    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    model = D2P3AFormer(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        learnable_queries=args.learnable_queries,
        input_shape=(args.input_h, args.input_w),
        eval=args.eval,
    )

    # weights
    weight_dict = {
        "hm": args.hm_weight,
        "reg": args.off_weight,
        "wh": args.wh_weight,
        "boxes": args.boxes_weight,
        "giou": args.giou_weight,
        "center_offset": args.ct_offset_weight,
        "tracking": args.tracking_weight,
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = None if args.eval else GenericLoss(args, weight_dict).to(device)
    postprocessors = {"bbox": PostProcess(args, valid_ids)}
    return model, criterion, postprocessors
