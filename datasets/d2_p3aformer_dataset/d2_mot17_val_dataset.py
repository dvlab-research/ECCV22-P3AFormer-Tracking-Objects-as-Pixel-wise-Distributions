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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

try:
    from datasets.d2_p3aformer_dataset.d2_generic_dataset_val import (
        GenericDataset_val,
    )
except:
    from datasets.d2_p3aformer_dataset.d2_generic_dataset_val import (
        GenericDataset_val,
    )
from detectron2.config import configurable


class MOT17_val(GenericDataset_val):
    num_classes = 1
    default_resolution = [640, 1088]
    max_objs = 300
    class_name = ["person"]
    cat_ids = {1: 1}

    @configurable
    def __init__(self, data_dir, split, input_w, input_h, output_w, output_h, private):
        if split == "test":
            img_dir = os.path.join(data_dir, "test")
        else:
            img_dir = os.path.join(data_dir, "train")
        ann_path = os.path.join(data_dir, "annotations", "{}.json").format(split)
        print(f"==> initializing MOT17 {split} data from ann_path {ann_path}.")
        self.is_mot17 = False
        self.images = None
        super(MOT17_val, self).__init__(
            input_w=input_w,
            input_h=input_h,
            output_w=output_w,
            output_h=output_h,
            split=split,
            ann_path=ann_path,
            img_dir=img_dir,
            private=private,
        )
        self.num_samples = len(self.video_list)
        print("Loaded {} {} videos.".format(split, self.num_samples))

    @classmethod
    def from_config(cls, cfg):
        input_h, input_w = (
            cfg.MODEL.DENSETRACK.DEFAULT_RESOLUTION[0],
            cfg.MODEL.DENSETRACK.DEFAULT_RESOLUTION[1],
        )
        output_h = input_h // cfg.MODEL.DENSETRACK.DOWN_RATIO
        output_w = input_w // cfg.MODEL.DENSETRACK.DOWN_RATIO
        ret = {
            "data_dir": cfg.INPUT.VAL_DATA_DIR,
            "split": cfg.INPUT.SPLIT,
            "input_w": input_w,
            "input_h": input_h,
            "output_w": output_w,
            "output_h": output_h,
            "private": cfg.TRACK.DENSETRACK.PRIVATE,
        }
        return ret

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def _save_results(self, records, fpath):
        with open(fpath, "w") as fid:
            for record in records:
                line = json.dumps(record) + "\n"
                fid.write(line)
        return fpath

    def convert_eval_format(self, all_bboxes):
        detections = []
        person_id = 1
        for image_id in all_bboxes:
            if type(all_bboxes[image_id]) != type({}):
                # newest format
                dtboxes = []
                for j in range(len(all_bboxes[image_id])):
                    item = all_bboxes[image_id][j]
                    if item["class"] != person_id:
                        continue
                    bbox = item["bbox"]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "tag": 1,
                        "box": bbox_out,
                        "score": float("{:.2f}".format(item["score"])),
                    }
                    dtboxes.append(detection)
            img_info = self.coco.loadImgs(ids=[image_id])[0]
            file_name = img_info["file_name"]
            detections.append({"ID": file_name[:-4], "dtboxes": dtboxes})
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        self._save_results(
            self.convert_eval_format(results),
            "{}/results_crowdhuman.odgt".format(save_dir),
        )

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        try:
            os.system(
                "python tools/crowdhuman_eval/demo.py "
                + "../data/crowdhuman/annotation_val.odgt "
                + "{}/results_crowdhuman.odgt".format(save_dir)
            )
        except:
            print("Crowdhuman evaluation not setup!")
