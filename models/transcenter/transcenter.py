## TransCenter: Transformers with Dense Queries for Multiple-Object Tracking
## Copyright Inria
## Year 2021
## Contact : yihong.xu@inria.fr
##
## TransCenter is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## TransCenter is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program, TransCenter.  If not, see <http://www.gnu.org/licenses/> and the LICENSE file.
##
##
## TransCenter has code derived from
## (1) 2020 fundamentalvision.(Apache License 2.0: https://github.com/fundamentalvision/Deformable-DETR)
## (2) 2020 Philipp Bergmann, Tim Meinhardt. (GNU General Public License v3.0 Licence: https://github.com/phil-bergmann/tracking_wo_bnw)
## (3) 2020 Facebook. (Apache License Version 2.0: https://github.com/facebookresearch/detr/)
## (4) 2020 Xingyi Zhou.(MIT License: https://github.com/xingyizhou/CenterTrack)
##
## TransCenter uses packages from
## (1) 2019 Charles Shang. (BSD 3-Clause Licence: https://github.com/CharlesShang/DCNv2)
## (2) 2017 NVIDIA CORPORATION. (Apache License, Version 2.0: https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)
## (3) 2019 Simon Niklaus. (GNU General Public License v3.0: https://github.com/sniklaus/pytorch-liteflownet)
## (4) 2018 Tak-Wai Hui. (Copyright (c), see details in the LICENSE file: https://github.com/twhui/LiteFlowNet)
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .transcenter_losses.utils import _sigmoid
from .transcenter_losses.losses import FastFocalLoss, RegWeightedL1Loss, loss_boxes
from util.p3aformer.p3aformer_misc import NestedTensor
from .transcenter_post_processing.decode import generic_decode
from .transcenter_post_processing.post_process import generic_post_process
from .transcenter_backbone import build_backbone
from .transcenter_deformable_transformer import build_deforamble_transformer
import copy
from .transcenter_dla import IDAUpV3
from torch import Tensor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class GenericLoss(torch.nn.Module):
    def __init__(self, opt, weight_dict):
        super(GenericLoss, self).__init__()
        self.crit = FastFocalLoss()
        self.crit_reg = RegWeightedL1Loss()
        self.opt = opt
        self.weight_dict = weight_dict

    def _sigmoid_output(self, output):
        if "hm" in output:
            output["hm"] = _sigmoid(output["hm"])
        return output

    def forward(self, outputs, batch):
        opt = self.opt
        regression_heads = ["reg", "wh", "tracking", "center_offset"]
        losses = {}

        outputs = self._sigmoid_output(outputs)

        for s in range(opt.dec_layers):
            if s < opt.dec_layers - 1:
                end_str = f"_{s}"
            else:
                end_str = ""

            # only 'hm' is use focal loss for heatmap regression. #
            if "hm" in outputs:
                losses["hm" + end_str] = (
                    self.crit(
                        outputs["hm"][s],
                        batch["hm"],
                        batch["ind"],
                        batch["mask"],
                        batch["cat"],
                    )
                    / opt.norm_factor
                )

            for head in regression_heads:
                if head in outputs:
                    # print(head)
                    losses[head + end_str] = (
                        self.crit_reg(
                            outputs[head][s],
                            batch[head + "_mask"],
                            batch["ind"],
                            batch[head],
                        )
                        / opt.norm_factor
                    )

            losses["boxes" + end_str], losses["giou" + end_str] = loss_boxes(
                outputs["boxes"][s], batch
            )
            losses["boxes" + end_str] /= opt.norm_factor
            losses["giou" + end_str] /= opt.norm_factor

        return losses


class DeformableDETR(nn.Module):
    """This is the Deformable DETR module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        aux_loss=True,
        learnable_queries=False,
        input_shape=(640, 1088),
        eval=True,
    ):
        """Initializes the model."""
        super().__init__()
        self.transformer = transformer
        self.eval_mode = eval
        hidden_dim = transformer.d_model

        self.ida_up = IDAUpV3(64, [256, 256, 256, 256], [2, 4, 8, 16])
        self.ida_up = _get_clones(self.ida_up, 2)

        """
        (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
        """
        # xyh #
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
        self.tracking = nn.Sequential(
            nn.Conv2d(
                129, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # init weights #
        # prior bias
        self.hm[-1].bias.data.fill_(-4.6)
        fill_fc_weights(self.reg)
        fill_fc_weights(self.wh)
        fill_fc_weights(self.ct_offset)
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

    def forward(
        self,
        samples: NestedTensor,
        pre_samples: NestedTensor,
        pre_hm: Tensor,
        features: Tensor = None,
        pos: Tensor = None,
        pre_features: Tensor = None,
        pre_pos: Tensor = None,
    ):
        assert isinstance(samples, NestedTensor)
        assert isinstance(pre_samples, NestedTensor)
        if features is None:
            features, pos = self.backbone(samples)

        srcs = []
        masks = []
        with torch.no_grad():
            if (pre_features is None and self.eval_mode) or not self.eval_mode:
                pre_features, pre_pos = self.backbone(pre_samples)
            pre_srcs = []
            pre_masks = []

        # xyh #
        for l, (feat, pre_feat) in enumerate(zip(features, pre_features)):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)

            # xyh pre
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

        hs = []
        pre_hs = []

        pre_hm_out = F.interpolate(
            pre_hm.float(), size=(pre_hm.shape[2] // 4, pre_hm.shape[3] // 4)
        )

        for hs_m, pre_hs_m in merged_hs:
            hs.append(hs_m)
            pre_hs.append(pre_hs_m)

        outputs_coords = []
        outputs_hms = []
        outputs_regs = []
        outputs_whs = []
        outputs_ct_offsets = []
        outputs_tracking = []

        for layer_lvl in range(len(hs)):

            hs[layer_lvl] = self.ida_up[0](hs[layer_lvl], 0, len(hs[layer_lvl]))[-1]
            pre_hs[layer_lvl] = self.ida_up[1](
                pre_hs[layer_lvl], 0, len(pre_hs[layer_lvl])
            )[-1]

            ct_offset = self.ct_offset(hs[layer_lvl])
            wh_head = self.wh(hs[layer_lvl])
            reg_head = self.reg(hs[layer_lvl])
            hm_head = self.hm(hs[layer_lvl])

            tracking_head = self.tracking(
                torch.cat(
                    [hs[layer_lvl].detach(), pre_hs[layer_lvl], pre_hm_out], dim=1
                )
            )

            outputs_whs.append(wh_head)
            outputs_ct_offsets.append(ct_offset)
            outputs_regs.append(reg_head)
            outputs_hms.append(hm_head)
            outputs_tracking.append(tracking_head)

            # b,2,h,w => b,4,h,w
            if not self.eval_mode:
                outputs_coords.append(torch.cat([reg_head + ct_offset, wh_head], dim=1))
        out = {
            "hm": torch.stack(outputs_hms),
            "wh": torch.stack(outputs_whs),
            "reg": torch.stack(outputs_regs),
            "center_offset": torch.stack(outputs_ct_offsets),
            "tracking": torch.stack(outputs_tracking),
        }
        if not self.eval_mode:
            out["boxes"] = torch.stack(outputs_coords)
        return out


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, args, valid_ids):
        self.args = args
        self._valid_ids = valid_ids
        print("valid_ids: ", self._valid_ids)
        print("self.args in post processor before:", self.args)
        super().__init__()

    def _sigmoid_output(self, output):
        if "hm" in output:
            output["hm"] = _sigmoid(output["hm"])
        return output

    @torch.no_grad()
    def forward(
        self,
        outputs,
        target_sizes,
        target_c=None,
        target_s=None,
        not_max_crop=False,
        filter_score=True,
    ):
        """Perform the computation"""

        # for map you dont need to filter
        if filter_score and self.args.eval:
            out_thresh = self.args.pre_thresh
        elif filter_score:
            out_thresh = self.args.out_thresh
        else:
            out_thresh = 0.0
        # get the output of last layer of transformer
        output = {k: v[-1].cpu() for k, v in outputs.items() if k != "boxes"}

        output = self._sigmoid_output(output)
        dets = generic_decode(output, K=self.args.K, opt=self.args)

        if target_c is None and target_s is None:
            target_c = []
            target_s = []
            for target_size in target_sizes:
                # get image centers
                target_size = target_size.cpu()
                c = np.array(
                    [target_size[1] / 2.0, target_size[0] / 2.0], dtype=np.float32
                )
                # get image size or max h or max w
                s = (
                    max(target_size[0], target_size[1]) * 1.0
                    if not self.args.not_max_crop
                    else np.array([target_size[1], target_size[0]], np.float32)
                )
                target_c.append(c)
                target_s.append(s)
        else:
            target_c = target_c.cpu().numpy()
            target_s = target_s.cpu().numpy()

        results = generic_post_process(
            self.args,
            dets,
            target_c,
            target_s,
            output["hm"].shape[2],
            output["hm"].shape[3],
            filter_by_scores=out_thresh,
        )
        coco_results = []
        for btch_idx in range(len(results)):
            boxes = []
            scores = []
            labels = []
            tracking = []
            pre_cts = []
            for det in results[btch_idx]:
                boxes.append(det["bbox"])
                scores.append(det["score"])
                labels.append(self._valid_ids[det["class"] - 1])
                tracking.append(det["tracking"])
                pre_cts.append(det["pre_cts"])
            if len(boxes) > 0:
                coco_results.append(
                    {
                        "scores": torch.as_tensor(scores).float(),
                        "labels": torch.as_tensor(labels).int(),
                        "boxes": torch.as_tensor(boxes).float(),
                        "tracking": torch.as_tensor(tracking).float(),
                        "pre_cts": torch.as_tensor(pre_cts).float(),
                    }
                )
            else:
                coco_results.append(
                    {
                        "scores": torch.zeros(0).float(),
                        "labels": torch.zeros(0).int(),
                        "boxes": torch.zeros(0, 4).float(),
                        "tracking": torch.zeros(0, 2).float(),
                        "pre_cts": torch.zeros(0, 2).float(),
                    }
                )
        return coco_results


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
    print("num_classes", num_classes)
    model = DeformableDETR(
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
