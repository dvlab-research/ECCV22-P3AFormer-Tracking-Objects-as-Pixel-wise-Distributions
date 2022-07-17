import numpy as np
from detectron2.config.config import configurable
import torch
from torch import nn
import torch
from torch import nn
import pdb
from torchvision.ops import boxes as box_ops
from .transcenter_losses.utils import _sigmoid
from .transcenter_post_processing.decode import generic_decode
from .transcenter_post_processing.post_process import generic_post_process


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api, used in the test time only!"""

    @configurable
    def __init__(self, eval, pre_thresh, out_thresh, k_element):
        super().__init__()
        self.eval = eval
        self.pre_thresh = pre_thresh
        self.out_thresh = out_thresh
        self.max_crop = True
        self.k_element = k_element
        valid_ids = [1]
        self._valid_ids = valid_ids

    @classmethod
    def from_config(cls, cfg):
        return {
            "eval": cfg.SOLVER.EVAL,
            "pre_thresh": cfg.TRACK.DENSETRACK.PRE_THRE,
            "out_thresh": cfg.TRACK.DENSETRACK.OUT_THRE,
            "k_element": cfg.TRACK.DENSETRACK.K_ELEMENT,
        }

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
        """
        Perform the computation
        """
        # for map you don't need to filter
        if filter_score and self.eval:
            out_thresh = self.pre_thresh
        elif filter_score:
            out_thresh = self.out_thresh
        else:
            out_thresh = 0.0

        # get the output of last layer of transformer
        output = {k: v[-1].cpu() for k, v in outputs.items() if k != "boxes"}
        output = self._sigmoid_output(output)
        dets = generic_decode(output, K=self.k_element)

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
                    if self.max_crop
                    else np.array([target_size[1], target_size[0]], np.float32)
                )
                target_c.append(c)
                target_s.append(s)
        else:
            target_c = target_c.cpu().numpy()
            target_s = target_s.cpu().numpy()
        results = generic_post_process(
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
                boxes.append(torch.tensor(det["bbox"]))
                scores.append(torch.tensor(det["score"]))
                labels.append(torch.tensor(self._valid_ids[det["class"] - 1]))
                if "tracking" in det:
                    tracking.append(det["tracking"])
                if "pre_cts" in det:
                    pre_cts.append(det["pre_cts"])
            if len(boxes) > 0:
                nms_bboxes_inds = box_ops.batched_nms(
                    torch.stack(boxes).float(),
                    torch.stack(scores),
                    torch.stack(labels),
                    0.4,
                )
                track_return = torch.as_tensor(tracking).float() if tracking else None
                pre_cts = torch.as_tensor(pre_cts).float() if pre_cts else None
                coco_results.append(
                    {
                        "scores": torch.as_tensor(scores)[nms_bboxes_inds].float(),
                        "labels": torch.as_tensor(labels)[nms_bboxes_inds].int(),
                        "boxes": torch.stack(boxes)[nms_bboxes_inds].float(),
                        "tracking": track_return,
                        "pre_cts": pre_cts,
                        "heatmap": output["hm"][btch_idx],
                    }
                )
            else:
                track_return = torch.zeros(0, 2).float() if tracking else None
                pre_cts = torch.zeros(0, 2).float() if pre_cts else None
                coco_results.append(
                    {
                        "scores": torch.zeros(0).float(),
                        "labels": torch.zeros(0).int(),
                        "boxes": torch.zeros(0, 4).float(),
                        "tracking": track_return,
                        "pre_cts": pre_cts,
                        "heatmap": output["hm"][btch_idx],
                    }
                )
        return coco_results
