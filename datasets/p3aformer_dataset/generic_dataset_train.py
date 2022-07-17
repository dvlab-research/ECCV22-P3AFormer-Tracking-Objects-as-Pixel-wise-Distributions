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

import numpy as np
import math
import json
import cv2
import os
from collections import defaultdict
import pycocotools.coco as coco
import torch.utils.data as data
import sys

curr_pth = os.path.abspath(__file__)
curr_pth = "/".join(curr_pth.split("/")[:-3])
sys.path.append(curr_pth)
from util.image import flip, color_aug, GaussianBlur
from util.image import get_affine_transform, affine_transform
from util.image import gaussian_radius, draw_umich_gaussian
import copy
from PIL import Image, ImageDraw, ImageFont
import time
from tqdm import tqdm
import pdb


class GenericDataset(data.Dataset):
    is_fusion_dataset = False
    default_resolution = None
    num_categories = None
    class_name = None
    # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
    # Not using 0 because 0 is used for don't care region and ignore loss.
    cat_ids = None
    max_objs = None
    rest_focal_length = 1200
    num_joints = 17
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    edges = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [4, 6],
        [3, 5],
        [5, 6],
        [5, 7],
        [7, 9],
        [6, 8],
        [8, 10],
        [6, 12],
        [5, 11],
        [11, 12],
        [12, 14],
        [14, 16],
        [11, 13],
        [13, 15],
    ]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)

    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
    _eig_vec = np.array(
        [
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938],
        ],
        dtype=np.float32,
    )
    ignore_val = 1
    nuscenes_att_range = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3, 4],
        3: [2, 3, 4],
        4: [2, 3, 4],
        5: [5, 6, 7],
        6: [5, 6, 7],
        7: [5, 6, 7],
    }

    sf = 0.2
    cf = 0.1

    def __init__(self, opt=None, split=None, ann_path=None, img_dir=None):
        super(GenericDataset, self).__init__()
        if opt is not None and split is not None:
            self.split = split
            self.opt = opt
            self._data_rng = np.random.RandomState(123)

        if ann_path is not None and img_dir is not None:
            print(
                "==> initializing {} data from {}, \n images from {} ...".format(
                    split, ann_path, img_dir
                )
            )
            self.coco = coco.COCO(ann_path)
            self.images = self.coco.getImgIds()

            if opt.tracking:
                if not ("videos" in self.coco.dataset):
                    self.fake_video_data()
                print("Creating video index!")
                self.video_to_images = defaultdict(list)
                for image in self.coco.dataset["images"]:
                    self.video_to_images[image["video_id"]].append(image)

            self.img_dir = img_dir

            if opt.cache_mode:
                self.cache = {}
                print("caching data into memory...")
                for tmp_im_id in tqdm(self.images):
                    img, anns, img_info, img_path = self._load_image_anns(
                        tmp_im_id, self.coco, self.img_dir
                    )
                    assert tmp_im_id not in self.cache.keys()
                    self.cache[tmp_im_id] = [img, anns, img_info, img_path]
            else:
                self.cache = {}

        self.blur_aug = GaussianBlur(kernel_size=11)

    def __getitem__(self, index):
        opt = self.opt
        img, anns, img_info, img_path, pad_img = self._load_data(index)
        img_blurred = False
        if self.opt.image_blur_aug and np.random.rand() < 0.1 and self.split == "train":
            img = self.blur_aug(img)
            img_blurred = True

        # get image height and width
        height, width = img.shape[0], img.shape[1]
        # get image centers
        c = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0], dtype=np.float32)
        # get image size or max h or max w
        s = (
            max(img.shape[0], img.shape[1]) * 1.0
            if not self.opt.not_max_crop
            else np.array([img.shape[1], img.shape[0]], np.float32)
        )
        aug_s, rot, flipped = 1, 0, 0
        if self.split == "train":
            # drift image centers, change image size with scale, rotate image with rot.
            c, aug_s, rot = self._get_aug_param(c, s, width, height)
            s = s * aug_s
            # random flip
            if np.random.random() < opt.flip:
                flipped = 1
                img = img[:, ::-1, :].copy()
                anns = self._flip_anns(anns, width)

        # we will reshape image to standard input shape, trans_input =transform for resizing gt to input size
        trans_input = get_affine_transform(c, s, rot, [opt.input_w, opt.input_h])
        # the output heatmap size != input size, trans_output = transform for resizing gt to output size
        trans_output = get_affine_transform(c, s, rot, [opt.output_w, opt.output_h])
        inp, padding_mask = self._get_input(img, trans_input, padding_mask=pad_img)
        # plot
        ret = {"image": inp, "pad_mask": padding_mask.astype(np.bool)}
        gt_det = {"bboxes": [], "scores": [], "clses": [], "cts": []}

        # get pre info, pre info has the same transform then current info
        pre_cts, pre_track_ids = None, None
        if opt.tracking:
            # randomly select a pre image with random interval
            (
                pre_image,
                pre_anns,
                frame_dist,
                pre_img_id,
                pre_pad_image,
            ) = self._load_pre_data(
                img_info["video_id"],
                img_info["frame_id"],
                img_info["sensor_id"] if "sensor_id" in img_info else 1,
            )

            if self.opt.image_blur_aug and img_blurred and self.split == "train":
                # print("blur image")
                pre_image = self.blur_aug(pre_image)

            if flipped:
                pre_image = pre_image[:, ::-1, :].copy()
                pre_anns = self._flip_anns(pre_anns, width)
            # if same_aug_pre and pre_img != curr_img, we use the same data aug for this pre image.
            if opt.same_aug_pre and frame_dist != 0:
                trans_input_pre = trans_input.copy()
                trans_output_pre = trans_output.copy()
            else:
                c_pre, aug_s_pre, _ = self._get_aug_param(
                    c.copy(), copy.deepcopy(s), width, height, disturb=True
                )
                s_pre = s * aug_s_pre
                trans_input_pre = get_affine_transform(
                    c_pre, s_pre, rot, [opt.input_w, opt.input_h]
                )
                trans_output_pre = get_affine_transform(
                    c_pre, s_pre, rot, [opt.output_w, opt.output_h]
                )

            # transform pre_image as standard input shape, todo warning pre_anns are not yet transformed
            pre_img, pre_padding_mask = self._get_input(
                pre_image, trans_input_pre, padding_mask=pre_pad_image
            )
            # pre_hm is of standard input shape, todo pre_cts is in the output image plane
            pre_hm, pre_cts, pre_track_ids = self._get_pre_dets(
                pre_anns, trans_input_pre, trans_output_pre
            )
            ret["pre_img"] = pre_img
            ret["pre_pad_mask"] = pre_padding_mask.astype(np.bool)
            if opt.pre_hm:
                ret["pre_hm"] = pre_hm

        ### init samples
        self._init_ret(ret, gt_det)
        num_objs = min(len(anns), self.max_objs)
        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(self.cat_ids[ann["category_id"]])
            if cls_id > self.opt.num_classes or cls_id <= -999:
                continue
            # get ground truth bbox in the output image plane,
            # bbox_amodal do not clip by ouput image size, bbox is clipped, todo !!!warning!!! the function performs cxcy2xyxy
            bbox, bbox_amodal = self._get_bbox_output(
                ann["bbox"], trans_output, height, width
            )
            if cls_id <= 0 or ("iscrowd" in ann and ann["iscrowd"] > 0):
                self._mask_ignore_or_crowd(ret, cls_id, bbox)
                # print('mask ignore or crowd.')
                continue

            # todo warning track_ids are ids at t-1
            self._add_instance(
                ret,
                gt_det,
                k,
                cls_id,
                bbox,
                bbox_amodal,
                ann,
                trans_output,
                aug_s,
                pre_cts,
                pre_track_ids,
            )

        if self.opt.debug > 0:
            gt_det = self._format_gt_det(gt_det)
            meta = {
                "c": c,
                "s": s,
                "gt_det": gt_det,
                "img_id": img_info["id"],
                "img_path": img_path,
                "flipped": flipped,
            }
            ret["meta"] = meta

        ret["c"] = c
        ret["s"] = np.asarray(s, dtype=np.float32)
        ret["image_id"] = self.images[index]
        ret["output_size"] = np.asarray([self.opt.output_h, self.opt.output_w])
        ret["orig_size"] = np.asarray([height, width])

        return ret

    def get_default_calib(self, width, height):
        calib = np.array(
            [
                [self.rest_focal_length, 0, width / 2, 0],
                [0, self.rest_focal_length, height / 2, 0],
                [0, 0, 1, 0],
            ]
        )
        return calib

    def _load_image_anns(self, img_id, coco, img_dir):
        img_info = coco.loadImgs(ids=[img_id])[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(img_dir, file_name)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        # bgr=> rgb
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, anns, img_info, img_path

    def _load_data(self, index):
        coco = self.coco
        img_dir = self.img_dir
        img_id = self.images[index]
        if img_id in self.cache.keys():
            img, anns, img_info, img_path = self.cache[img_id]
        else:
            img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)
        # padding before affine warping to prevent cropping
        h, w, c = img.shape
        target_ratio = 1.0 * self.opt.input_w / self.opt.input_h
        if 1.0 * w / h < target_ratio:
            new_w = int(target_ratio * h)
            new_img = np.zeros((h, new_w, c)).astype(img.dtype)
            new_img[:, :w, :] = img
            if "width" in img_info.keys():
                img_info["width"] = new_w

        else:
            new_img = img

        return new_img, anns, img_info, img_path, np.ones_like(img)

    def _load_pre_data(self, video_id, frame_id, sensor_id=1):
        img_infos = self.video_to_images[video_id]
        # If training, random sample nearby frames as the "previous" frame
        # If testing, get the exact prevous frame
        if "train" in self.split:
            img_ids = [
                (img_info["id"], img_info["frame_id"])
                for img_info in img_infos
                if abs(img_info["frame_id"] - frame_id) < self.opt.max_frame_dist
                and (
                    not ("sensor_id" in img_info) or img_info["sensor_id"] == sensor_id
                )
            ]
        else:
            img_ids = [
                (img_info["id"], img_info["frame_id"])
                for img_info in img_infos
                if (img_info["frame_id"] - frame_id) == -1
                and (
                    not ("sensor_id" in img_info) or img_info["sensor_id"] == sensor_id
                )
            ]
            if len(img_ids) == 0:
                img_ids = [
                    (img_info["id"], img_info["frame_id"])
                    for img_info in img_infos
                    if (img_info["frame_id"] - frame_id) == 0
                    and (
                        not ("sensor_id" in img_info)
                        or img_info["sensor_id"] == sensor_id
                    )
                ]
        rand_id = np.random.choice(len(img_ids))

        img_id, pre_frame_id = img_ids[rand_id]
        frame_dist = abs(frame_id - pre_frame_id)
        # print(frame_dist)
        if img_id in self.cache.keys():
            img, anns, _, _ = self.cache[img_id]
        else:
            img, anns, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)

        # padding before affine warping to prevent cropping
        h, w, c = img.shape
        target_ratio = 1.0 * self.opt.input_w / self.opt.input_h
        if 1.0 * w / h < target_ratio:
            new_w = int(target_ratio * h)
            new_img = np.zeros((h, new_w, c)).astype(img.dtype)
            new_img[:, :w, :] = img

        else:
            new_img = img
        return new_img, anns, frame_dist, img_id, np.ones_like(img)

    def _get_pre_dets(self, anns, trans_input, trans_output):
        hm_h, hm_w = self.opt.input_h, self.opt.input_w
        down_ratio = self.opt.down_ratio
        trans = trans_input
        reutrn_hm = self.opt.pre_hm
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
        pre_cts, track_ids = [], []
        for ann in anns:
            cls_id = int(self.cat_ids[ann["category_id"]])
            if (
                cls_id > self.opt.num_classes
                or cls_id <= -99
                or ("iscrowd" in ann and ann["iscrowd"] > 0)
            ):
                continue
            bbox = self._coco_box_to_bbox(ann["bbox"])
            # from original input image size to standard input size using draw_umich_gaussian
            bbox[:2] = affine_transform(bbox[:2], trans)
            bbox[2:] = affine_transform(bbox[2:], trans)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            max_rad = 1
            # draw gt heatmap with
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                max_rad = max(max_rad, radius)
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
                )
                ct0 = ct.copy()
                conf = 1
                # add some noise to ground-truth pre info
                ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
                ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
                conf = 1 if np.random.random() > self.opt.lost_disturb else 0

                ct_int = ct.astype(np.int32)
                if conf == 0:
                    pre_cts.append(ct / down_ratio)
                else:
                    pre_cts.append(ct0 / down_ratio)

                # conf == 0, lost hm, FN
                track_ids.append(ann["track_id"] if "track_id" in ann else -1)
                if reutrn_hm:
                    draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

                # false positives disturb
                if np.random.random() < self.opt.fp_disturb and reutrn_hm:
                    ct2 = ct0.copy()
                    # Hard code heatmap disturb ratio, haven't tried other numbers.
                    ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                    ct2[1] = ct2[1] + np.random.randn() * 0.05 * h
                    ct2_int = ct2.astype(np.int32)
                    draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)
        return pre_hm, pre_cts, track_ids

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_aug_param(self, c, s, width, height, disturb=False):
        if (not self.opt.not_rand_crop) and not disturb:
            sf = self.sf
            cf = self.cf

            if type(s) == float or type(s) == np.float64 or type(s) == np.float32:
                s = [s, s]
            c[0] += s[0] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s[1] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        else:
            sf = self.opt.scale
            cf = self.opt.shift
            # print(s)
            if type(s) == float or type(s) == np.float64 or type(s) == np.float32:
                s = [s, s]
            c[0] += s[0] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s[1] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.opt.aug_rot:
            print("random rotate is activated.")
            rf = self.opt.rotate
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        else:
            rot = 0

        return c, aug_s, rot

    def _flip_anns(self, anns, width):
        for k in range(len(anns)):
            bbox = anns[k]["bbox"]
            anns[k]["bbox"] = [width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

            if "hps" in self.opt.heads and "keypoints" in anns[k]:
                keypoints = np.array(anns[k]["keypoints"], dtype=np.float32).reshape(
                    self.num_joints, 3
                )
                keypoints[:, 0] = width - keypoints[:, 0] - 1
                for e in self.flip_idx:
                    keypoints[e[0]], keypoints[e[1]] = (
                        keypoints[e[1]].copy(),
                        keypoints[e[0]].copy(),
                    )
                anns[k]["keypoints"] = keypoints.reshape(-1).tolist()

            if "rot" in self.opt.heads and "alpha" in anns[k]:
                anns[k]["alpha"] = (
                    np.pi - anns[k]["alpha"]
                    if anns[k]["alpha"] > 0
                    else -np.pi - anns[k]["alpha"]
                )

            if "amodel_offset" in self.opt.heads and "amodel_center" in anns[k]:
                anns[k]["amodel_center"][0] = width - anns[k]["amodel_center"][0] - 1
        return anns

    def _get_input(self, img, trans_input, padding_mask=None):
        img = img.copy()
        if padding_mask is None:
            padding_mask = np.ones_like(img)
        inp = cv2.warpAffine(
            img,
            trans_input,
            (self.opt.input_w, self.opt.input_h),
            flags=cv2.INTER_LINEAR,
        )

        # to mask = 1 (padding part), not to mask = 0
        affine_padding_mask = cv2.warpAffine(
            padding_mask,
            trans_input,
            (self.opt.input_w, self.opt.input_h),
            flags=cv2.INTER_LINEAR,
        )
        affine_padding_mask = affine_padding_mask[:, :, 0]
        affine_padding_mask[affine_padding_mask > 0] = 1

        inp = inp.astype(np.float32) / 255.0
        if (
            self.split == "train"
            and not self.opt.no_color_aug
            and np.random.rand() < 0.2
        ):
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp, 1 - affine_padding_mask

    def _init_ret(self, ret, gt_det):
        max_objs = self.max_objs * self.opt.dense_reg
        ret["hm"] = np.zeros(
            (self.opt.num_classes, self.opt.output_h, self.opt.output_w), np.float32
        )
        ret["ind"] = np.zeros((max_objs), dtype=np.int64)
        ret["cat"] = np.zeros((max_objs), dtype=np.int64)
        ret["mask"] = np.zeros((max_objs), dtype=np.float32)
        # xyh #
        ret["boxes"] = np.zeros((max_objs, 4), dtype=np.float32)
        ret["boxes_mask"] = np.zeros((max_objs), dtype=np.float32)

        ret["center_offset"] = np.zeros((max_objs, 2), dtype=np.float32)

        regression_head_dims = {
            "reg": 2,
            "wh": 2,
            "tracking": 2,
            "ltrb": 4,
            "ltrb_amodal": 4,
            "nuscenes_att": 8,
            "velocity": 3,
            "hps": self.num_joints * 2,
            "dep": 1,
            "dim": 3,
            "amodel_offset": 2,
            "center_offset": 2,
        }

        for head in regression_head_dims:
            if head in self.opt.heads:
                ret[head] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32
                )
                ret[head + "_mask"] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32
                )
                gt_det[head] = []

        if "hm_hp" in self.opt.heads:
            num_joints = self.num_joints
            ret["hm_hp"] = np.zeros(
                (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32
            )
            ret["hm_hp_mask"] = np.zeros((max_objs * num_joints), dtype=np.float32)
            ret["hp_offset"] = np.zeros((max_objs * num_joints, 2), dtype=np.float32)
            ret["hp_ind"] = np.zeros((max_objs * num_joints), dtype=np.int64)
            ret["hp_offset_mask"] = np.zeros(
                (max_objs * num_joints, 2), dtype=np.float32
            )
            ret["joint"] = np.zeros((max_objs * num_joints), dtype=np.int64)

        if "rot" in self.opt.heads:
            ret["rotbin"] = np.zeros((max_objs, 2), dtype=np.int64)
            ret["rotres"] = np.zeros((max_objs, 2), dtype=np.float32)
            ret["rot_mask"] = np.zeros((max_objs), dtype=np.float32)
            gt_det.update({"rot": []})

    def _get_calib(self, img_info, width, height):
        if "calib" in img_info:
            calib = np.array(img_info["calib"], dtype=np.float32)
        else:
            calib = np.array(
                [
                    [self.rest_focal_length, 0, width / 2, 0],
                    [0, self.rest_focal_length, height / 2, 0],
                    [0, 0, 1, 0],
                ]
            )
        return calib

    def _ignore_region(self, region, ignore_val=1):
        np.maximum(region, ignore_val, out=region)

    def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
        # mask out crowd region, only rectangular mask is supported
        if cls_id == 0:  # ignore all classes
            self._ignore_region(
                ret["hm"][
                    :, int(bbox[1]) : int(bbox[3]) + 1, int(bbox[0]) : int(bbox[2]) + 1
                ]
            )
        else:
            # mask out one specific class
            self._ignore_region(
                ret["hm"][
                    abs(cls_id) - 1,
                    int(bbox[1]) : int(bbox[3]) + 1,
                    int(bbox[0]) : int(bbox[2]) + 1,
                ]
            )
        if ("hm_hp" in ret) and cls_id <= 1:
            self._ignore_region(
                ret["hm_hp"][
                    :, int(bbox[1]) : int(bbox[3]) + 1, int(bbox[0]) : int(bbox[2]) + 1
                ]
            )

    def _coco_box_to_bbox(self, box):
        bbox = np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32
        )
        return bbox

    def _get_bbox_output(self, bbox, trans_output, height, width):
        bbox = self._coco_box_to_bbox(bbox).copy()

        rect = np.array(
            [
                [bbox[0], bbox[1]],
                [bbox[0], bbox[3]],
                [bbox[2], bbox[3]],
                [bbox[2], bbox[1]],
            ],
            dtype=np.float32,
        )
        for t in range(4):
            rect[t] = affine_transform(rect[t], trans_output)
        bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
        bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

        bbox_amodal = copy.deepcopy(bbox)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
        return bbox, bbox_amodal

    def _add_instance(
        self,
        ret,
        gt_det,
        k,
        cls_id,
        bbox,
        bbox_amodal,
        ann,
        trans_output,
        aug_s,
        pre_cts=None,
        pre_track_ids=None,
    ):

        # box is in the output image plane, add it to gt heatmap
        h, w = bbox_amodal[3] - bbox_amodal[1], bbox_amodal[2] - bbox_amodal[0]
        h_clip, w_clip = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h_clip <= 0 or w_clip <= 0:
            return
        # print(k)
        radius = gaussian_radius((math.ceil(h_clip), math.ceil(w_clip)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
        )
        # int(ct)
        ct_int = ct.astype(np.int32)

        # 'cat': categories of shape [num_objects], recording the cat id.
        ret["cat"][k] = cls_id - 1
        # 'mask': mask of shape [num_objects], if mask == 1, to train, if mask == 0, not to train.
        ret["mask"][k] = 1
        if "wh" in ret:
            # 'wh' = box_amodal size,of shape [num_objects, 2]
            ret["wh"][k] = 1.0 * w, 1.0 * h
            ret["wh_mask"][k] = 1
        # 'ind' of shape [num_objects],
        # indicating the position of the object = y*W_output + x in a heatmap of shape [out_h, out_w] #todo warning CT_INT
        ret["ind"][k] = ct_int[1] * self.opt.output_w + ct_int[0]
        # the .xxx part of the kpts
        ret["reg"][k] = ct - ct_int
        ret["reg_mask"][k] = 1

        # center_offset
        ret["center_offset"][k] = (
            0.5 * (bbox_amodal[0] + bbox_amodal[2]) - ct[0],
            0.5 * (bbox_amodal[1] + bbox_amodal[3]) - ct[1],
        )

        ret["center_offset_mask"][k] = 1

        # ad pts to ground-truth heatmap
        # print("ct_int", ct_int)

        draw_umich_gaussian(ret["hm"][cls_id - 1], ct_int, radius)

        gt_det["bboxes"].append(
            np.array(
                [ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2],
                dtype=np.float32,
            )
        )
        # cx, cy, w, h
        # clipped box
        # ret['boxes'][k] = np.asarray([ct[0], ct[1], w, h], dtype=np.float32)
        ret["boxes"][k] = np.asarray(
            [
                0.5 * (bbox_amodal[0] + bbox_amodal[2]),
                0.5 * (bbox_amodal[1] + bbox_amodal[3]),
                (bbox_amodal[2] - bbox_amodal[0]),
                (bbox_amodal[3] - bbox_amodal[1]),
            ],
            dtype=np.float32,
        )

        # cx, cy, w, h / output size
        ret["boxes"][k][0::2] /= self.opt.output_w
        ret["boxes"][k][1::2] /= self.opt.output_h
        ret["boxes_mask"][k] = 1
        gt_det["scores"].append(1)
        gt_det["clses"].append(cls_id - 1)
        gt_det["cts"].append(ct)

        if "tracking" in self.opt.heads:
            # if 'tracking' we produce ground-truth offset heatmap
            # if curr track id exists in pre track ids
            if ann["track_id"] in pre_track_ids:
                # get pre center pos
                pre_ct = pre_cts[pre_track_ids.index(ann["track_id"])]
                ret["tracking_mask"][k] = 1
                # todo 'tracking' of shape [# current objects, 2], be careful pre_ct (float) - CT_INT (the int part)
                # predict(ct_int) + ret['tracking'][k] = pre_ct (bring you to pre center)
                ret["tracking"][k] = pre_ct - ct_int
                gt_det["tracking"].append(ret["tracking"][k])
            else:
                gt_det["tracking"].append(np.zeros(2, np.float32))

    def _format_gt_det(self, gt_det):
        if len(gt_det["scores"]) == 0:
            gt_det = {
                "bboxes": np.array([[0, 0, 1, 1]], dtype=np.float32),
                "scores": np.array([1], dtype=np.float32),
                "clses": np.array([0], dtype=np.float32),
                "cts": np.array([[0, 0]], dtype=np.float32),
                "pre_cts": np.array([[0, 0]], dtype=np.float32),
                "tracking": np.array([[0, 0]], dtype=np.float32),
                "bboxes_amodal": np.array([[0, 0]], dtype=np.float32),
                "hps": np.zeros((1, 17, 2), dtype=np.float32),
            }
        gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
        return gt_det

    def fake_video_data(self):
        self.coco.dataset["videos"] = []
        for i in range(len(self.coco.dataset["images"])):
            img_id = self.coco.dataset["images"][i]["id"]
            self.coco.dataset["images"][i]["video_id"] = img_id
            self.coco.dataset["images"][i]["frame_id"] = 1
            self.coco.dataset["videos"].append({"id": img_id})

        if not ("annotations" in self.coco.dataset):
            return

        for i in range(len(self.coco.dataset["annotations"])):
            self.coco.dataset["annotations"][i]["track_id"] = i + 1
