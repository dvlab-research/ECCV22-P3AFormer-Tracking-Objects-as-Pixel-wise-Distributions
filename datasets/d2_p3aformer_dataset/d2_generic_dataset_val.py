from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import OrderedDict

import numpy as np
import cv2
import os
import torch
from collections import defaultdict, OrderedDict
import pycocotools.coco as coco
import torch.utils.data as data
import sys

curr_pth = os.path.abspath(__file__)
curr_pth = "/".join(curr_pth.split("/")[:-3])
sys.path.append(curr_pth)
from util.image import get_affine_transform, affine_transform
import copy
from tqdm import tqdm
from util.p3aformer.p3aformer_misc import read_MOT17det
import pdb


class GenericDataset_val(data.Dataset):
    default_resolution = None
    num_categories = None
    class_name = None
    # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
    # Not using 0 because 0 is used for don't care region and ignore loss.
    cat_ids = None
    max_objs = None
    dets_path = "det/det.txt"
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __init__(
        self,
        input_w=None,
        input_h=None,
        output_w=None,
        output_h=None,
        split=None,
        ann_path=None,
        img_dir=None,
        private=True,
    ):
        super(GenericDataset_val, self).__init__()
        self.input_w = input_w
        self.input_h = input_h
        self.output_w = output_w
        self.output_h = output_h
        print(
            "==> initializing {} data from {}, \n images from {} ...".format(
                split, ann_path, img_dir
            )
        )
        self.coco = coco.COCO(ann_path)
        self.images = self.coco.getImgIds()
        self.img_dir = img_dir

        print("Creating video index!")
        self.video_to_images = defaultdict(list)
        self.VidtoVname = {}
        self.VidPubDet = {}
        for v_info in self.coco.dataset["videos"]:
            # if private: #todo: validate these lines.
            #     if "SDP" not in v_info['file_name'] and "MOT17" in v_info['file_name']:
            #         continue
            self.VidtoVname[v_info["id"]] = v_info["file_name"]
            self.VidPubDet[v_info["id"]] = read_MOT17det(
                os.path.join(self.img_dir, v_info["file_name"], self.dets_path)
            )

        for image in self.coco.dataset["images"]:
            if image["video_id"] not in self.VidtoVname.keys():
                continue
            self.video_to_images[self.VidtoVname[image["video_id"]]].append(image)

        self.video_list = list(self.VidtoVname.keys())
        self.cache = {}

    def __getitem__(self, v_index):
        v_id = self.video_list[v_index]
        video_name = self.VidtoVname[v_id]
        rets = OrderedDict()
        for image_info in self.video_to_images[video_name]:
            img_id = image_info["id"]
            frame_id = int(image_info["file_name"].split("/")[-1][:-4])
            ret = {"img_id": img_id, "frame_id": frame_id}
            rets[image_info["file_name"].split("/")[-1]] = ret
        rets = OrderedDict(sorted(rets.items()))
        return rets, video_name

    def _load_pre_data(self, video_id, frame_id):
        img_infos = self.video_to_images[self.VidtoVname[video_id]]
        # If training, random sample nearby frames as the "previous" frame
        # If testing, get the exact prevous frame
        img_ids = [
            (img_info["id"], img_info["frame_id"])
            for img_info in img_infos
            if (img_info["frame_id"] - frame_id) == -1
        ]
        if len(img_ids) == 0:
            img_ids = [
                (img_info["id"], img_info["frame_id"])
                for img_info in img_infos
                if (img_info["frame_id"] - frame_id) == 0
            ]

        rand_id = 0
        assert len(img_ids) == 1
        img_id, pre_frame_id = img_ids[rand_id]
        frame_dist = abs(frame_id - pre_frame_id)
        assert frame_dist in [0, 1]
        if img_id in self.cache.keys():
            img, _, _, _ = self.cache[img_id]
        else:
            img, _, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)

        # padding before affine warping to prevent cropping
        h, w, c = img.shape
        target_ratio = 1.0 * self.input_w / self.input_h
        if 1.0 * w / h < target_ratio:
            new_w = int(target_ratio * h)
            new_img = np.zeros((h, new_w, c)).astype(img.dtype)
            new_img[:, :w, :] = img

        else:
            new_img = img
        return new_img, None, frame_dist, img_id, np.ones_like(img)

    def _load_image_anns(self, img_id, coco, img_dir):
        img_info = coco.loadImgs(ids=[img_id])[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(img_dir, file_name)
        assert os.path.exists(img_path), f"{img_path} does not exist!"
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, None, img_info, img_path

    def load_data(self, idx, img_id, visualizer):
        coco = self.coco
        img_dir = self.img_dir
        if img_id in self.cache.keys():
            img, anns, img_info, img_path = self.cache[img_id]
        else:
            img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)
        # padding before affine warping to prevent cropping
        h, w, c = img.shape
        target_ratio = 1.0 * self.input_w / self.input_h
        if 1.0 * w / h < target_ratio:
            new_w = int(target_ratio * h)
            new_img = np.zeros((h, new_w, c)).astype(img.dtype)
            new_img[:, :w, :] = img
        else:
            new_img = img
        img = new_img
        pad_mask = np.ones_like(img)
        if visualizer:
            visualizer.add_img(img, img_id=idx)
        height, width = img.shape[0], img.shape[1]
        trans_input, c, s = None, None, None
        if c is None:
            # get image centers
            c = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0], dtype=np.float32)
            # get image size or max h or max w, adopt max crop
            s = max(img.shape[0], img.shape[1]) * 1.0

        aug_s, rot, flipped = 1, 0, 0
        if trans_input is None:
            # resize input
            trans_input = get_affine_transform(c, s, rot, [self.input_w, self.input_h])
        inp, padding_mask = self.resize_input(img, trans_input, padding_mask=pad_mask)
        # load a pre image with random interval #  # TODO: validate this comment
        (
            pre_image,
            _,
            frame_dist,
            pre_img_id,
            pre_pad_mask,
        ) = self._load_pre_data(img_info["video_id"], img_info["frame_id"])
        pre_inp, pre_padding_mask = self.resize_input(
            pre_image, trans_input, padding_mask=pre_pad_mask
        )
        batch = {
            "image": torch.from_numpy(inp).unsqueeze_(0).cuda(),
            "pad_mask": torch.from_numpy(padding_mask.astype(np.bool))
            .unsqueeze_(0)
            .cuda(),
            "pre_image": torch.from_numpy(pre_inp).unsqueeze_(0).cuda(),
            "pre_pad_mask": torch.from_numpy(pre_padding_mask.astype(np.bool))
            .unsqueeze_(0)
            .cuda(),
            "trans_input": torch.from_numpy(trans_input).unsqueeze_(0).cuda(),
            "frame_dist": frame_dist,
            "orig_size": torch.from_numpy(np.asarray([height, width]))
            .unsqueeze_(0)
            .cuda(),
            "dets": None,
            "orig_img": torch.from_numpy(
                np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)
            ).unsqueeze_(0),
        }
        return batch

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def resize_input(self, img, trans_input, padding_mask=None):
        img = img.copy()
        if padding_mask is None:
            padding_mask = np.ones_like(img)
        cv2.imwrite("before_trans.jpg", img)
        inp = cv2.warpAffine(
            img, trans_input, (self.input_w, self.input_h), flags=cv2.INTER_LINEAR
        )
        cv2.imwrite("after_trans.jpg", inp)
        affine_padding_mask = cv2.warpAffine(
            padding_mask,
            trans_input,
            (self.input_w, self.input_h),
            flags=cv2.INTER_LINEAR,
        )
        affine_padding_mask = affine_padding_mask[:, :, 0]
        affine_padding_mask[affine_padding_mask > 0] = 1

        inp = inp.astype(np.float32) / 255.0
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp, 1 - affine_padding_mask

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
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        return bbox, bbox_amodal

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
