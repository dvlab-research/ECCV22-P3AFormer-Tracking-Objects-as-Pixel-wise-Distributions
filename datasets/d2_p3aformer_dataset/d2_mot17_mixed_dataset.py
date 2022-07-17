# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import numpy as np
import torch
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
import os
import numpy as np
import math
import json
import cv2
import os
import pycocotools.coco as coco
import pdb
from util.image import flip, color_aug, GaussianBlur
from util.image import get_affine_transform, affine_transform
from util.image import gaussian_radius, draw_umich_gaussian
from tools.visualization_tool import Visualizer
import copy
from detectron2.data.datasets import load_coco_json


def mot17_mixed_dataset_function(cfg):
    ann_path = os.path.join(cfg.INPUT.DATA_DIR, "annotations", "{}.json").format(
        "train"
    )  # consider validation
    img_folder = os.path.join(cfg.INPUT.DATA_DIR)
    dataset_list = load_coco_json(
        ann_path,
        image_root=img_folder,
    )
    return dataset_list


def configured_mot17_test_dataset_function(cfg):
    def one_f():
        ann_path = os.path.join(cfg.INPUT.VAL_DATA_DIR, "annotations", "train.json")
        img_folder = os.path.join(cfg.INPUT.VAL_DATA_DIR)
        dataset_list = load_coco_json(
            ann_path,
            image_root=img_folder,
        )
        return dataset_list

    return one_f


class MOT17DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by p3aformer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        *,
        is_train=True,
        tracking=False,
        down_ratio=4,
        same_aug_pre=True,
        pre_hm=False,
        data_dir="",
        heads=[],
        device="",
        visualizer: Visualizer = None,
        default_resolution=[],
        augs=[],
    ):
        self.split = "train"  # used in training only.
        self.num_classes = 1
        self.num_joints = 17
        self.visualizer = visualizer
        self.default_resolution = default_resolution
        self.max_objs = 300
        self.cat_ids = {1: 1}
        self.flip_idx = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
        ]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array(
            [
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938],
            ],
            dtype=np.float32,
        )
        self.ignore_val = 1
        self.sf = 0.2
        self.cf = 0.1
        self.color_aug = True
        self.image_blur_aug = True
        self.max_crop = True
        self.rand_crop = True
        self.rand_crop_scale = 0.05
        self.rand_crop_shift = 0.05
        self.flip_ratio = 0.5
        self.hm_disturb = 0.05
        self.fp_disturb = 0.1
        self.lost_disturb = 0.4
        # options from cfgs
        self.is_train = is_train
        self.tracking = tracking
        self.down_ratio = down_ratio
        self.same_aug_pre = same_aug_pre
        self.input_h, self.input_w = (
            self.default_resolution[0],
            self.default_resolution[1],
        )
        self.output_h = self.input_h // self.down_ratio
        self.output_w = self.input_w // self.down_ratio
        self.input_res = max(self.input_h, self.input_w)
        self.output_res = max(self.output_h, self.output_w)
        self.pre_hm = pre_hm
        self._data_rng = np.random.RandomState(123)
        self.blur_aug = GaussianBlur(kernel_size=11)
        self.data_dir = data_dir
        ann_path = os.path.join(data_dir, "annotations", "{}.json").format("train")
        self.coco = coco.COCO(ann_path)
        self.images = self.coco.getImgIds()
        self.heads = heads
        self.device = device
        self.augs = augs

    @classmethod
    def from_config(cls, cfg, is_train=True):
        visualizer = Visualizer() if cfg.TRACK.VIS else None
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())
        ret = {
            "is_train": is_train,
            "tracking": cfg.MODEL.DENSETRACK.TRACKING,
            "down_ratio": cfg.MODEL.DENSETRACK.DOWN_RATIO,
            "same_aug_pre": cfg.MODEL.DENSETRACK.SAME_AUG_PRE,
            "pre_hm": cfg.MODEL.DENSETRACK.PRE_HM,
            "data_dir": cfg.INPUT.DATA_DIR,
            "heads": cfg.MODEL.DENSETRACK.HEADS,
            "device": cfg.MODEL.DEVICE,
            "default_resolution": cfg.MODEL.DENSETRACK.DEFAULT_RESOLUTION,
            "visualizer": visualizer,
            "augs": augs,
        }
        return ret

    def vis_img(self, img, img_id, bbox_xyxy, toxyxy=False, heatmap=None):
        if self.visualizer:
            self.visualizer.add_img(img, img_id)
            for ann in bbox_xyxy:
                if toxyxy:
                    bbox = [
                        ann[0],
                        ann[1],
                        ann[2] + ann[0],
                        ann[3] + ann[1],
                    ]
                else:
                    bbox = [
                        ann[0],
                        ann[1],
                        ann[2],
                        ann[3],
                    ]
                self.visualizer.add_coco_bbox(
                    bbox, 0, conf=0.0, add_txt="", img_id=img_id
                )
            if heatmap is not None:
                self.visualizer.add_heatmap(heatmap, img_id, 0.1)
            self.visualizer.save_img(img_id, "./")

    def fetch_bboxes_from_annotations(self, anns):
        """
        Fetch bboxes (XYXY, ABS) from anns.
        """
        bboxes = []
        for ann in anns:
            one_box = ann["bbox"]
            one_box[2] += one_box[0]
            one_box[3] += one_box[1]
            bboxes.append(one_box)
        return torch.tensor(bboxes)

    def apply_resize_aug(self, aug_input):
        aug_input = T.AugInput(aug_input.image, boxes=aug_input.boxes)
        scale_aug = [
            T.ScaleTransform(
                h=aug_input.image.shape[0],
                w=aug_input.image.shape[1],
                new_h=self.input_h,
                new_w=self.input_w,
                interp="bilinear",
            )
        ]
        aug_input, _ = T.apply_transform_gens(scale_aug, aug_input)
        return aug_input

    def image_normalize(self, image_array):
        assert (
            image_array.shape[2] == 3
        ), "assume image to be HWC in the normalize function"
        image_array = image_array.astype(np.float32) / 255.0
        return (image_array - self.mean) / self.std

    def scale_bbox_to_output_space(
        self, xyxy_bbox, width, height, output_w, output_h, clip=True
    ):
        def clip_by_thre(value, up_thre, low_thre=0):
            if value > up_thre - 1:
                return up_thre - 1
            elif value < low_thre:
                return 0
            else:
                return value

        new_bbox = copy.deepcopy(xyxy_bbox)
        if not (xyxy_bbox[2] >= xyxy_bbox[0] and xyxy_bbox[3] >= xyxy_bbox[1]):
            print("bbox not xyxy format!")
            pdb.set_trace()
        new_bbox[0] = new_bbox[0] / width * output_w
        new_bbox[1] = new_bbox[1] / height * output_h
        new_bbox[2] = new_bbox[2] / width * output_w
        new_bbox[3] = new_bbox[3] / height * output_h
        if clip:
            new_bbox[0] = clip_by_thre(new_bbox[0], output_w)
            new_bbox[1] = clip_by_thre(new_bbox[1], output_h)
            new_bbox[2] = clip_by_thre(new_bbox[2], output_w)
            new_bbox[3] = clip_by_thre(new_bbox[3], output_h)
        return new_bbox

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that P3AFormer in detectron2 accept
        """
        assert self.is_train, "MOT17DatasetMapper should only be used for training!"
        img_id = dataset_dict["image_id"]
        img, anns, img_info, img_path, pad_img = self._load_data_given_img_id(img_id)
        (
            inp,
            padding_mask,
            img_blurred,
            flipped,
            trans_input,
            trans_output,
            c,
            aug_s,
            s,
            rot,
            width,
            height,
        ) = self._img_preprocess(anns, img, pad_img)
        ret = {"image": inp, "pad_mask": padding_mask.astype(np.bool)}
        gt_det = {"bboxes": [], "scores": [], "clses": [], "cts": []}
        # get pre info, pre info has the same transform then current info
        pre_cts, pre_track_ids = None, None
        if self.tracking:
            # load pre data
            pre_img_id = (
                img_id
                if (
                    img_info["prev_image_id"] == -1
                    or img_info["prev_image_id"] not in self.images
                )
                else img_info["prev_image_id"]
            )
            (
                pre_image,
                pre_anns,
                pre_img_info,
                pre_img_path,
                pre_pad_image,
            ) = self._load_data_given_img_id(pre_img_id)
            (
                pre_img,
                pre_padding_mask,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = self._img_preprocess(pre_anns, pre_image, pre_pad_image)
            frame_dist = abs(int(pre_img_id) - int(img_id))
            # TODO: validate this effect
            if self.same_aug_pre and frame_dist != 0:
                trans_input_pre = trans_input.copy()
            else:
                c_pre, aug_s_pre, _ = self._get_aug_param(
                    c.copy(), copy.deepcopy(s), width, height, disturb=True
                )
                s_pre = s * aug_s_pre
                trans_input_pre = get_affine_transform(
                    c_pre, s_pre, rot, [self.input_w, self.input_h]
                )

            # pre_hm is of standard input shape, todo pre_cts is in the output image plane
            pre_hm, pre_cts, pre_track_ids = self._get_pre_dets(
                pre_anns, trans_input_pre
            )
            ret.update(
                {"pre_img": pre_img, "pre_pad_mask": pre_padding_mask.astype(np.bool)}
            )
            if self.pre_hm:
                ret["pre_hm"] = pre_hm

        ### init samples
        self._init_ret(ret, gt_det)
        num_objs = min(len(anns), self.max_objs)
        vis_bbox = []
        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(self.cat_ids[ann["category_id"]])
            if cls_id > self.num_classes or cls_id <= -999:
                continue
            # get ground truth bbox in the output image plane,
            # bbox_amodal do not clip by ouput image size, bbox is clipped, todo !!!warning!!! the function performs cxcy2xyxy
            bbox, bbox_amodal = self._get_bbox_output(
                ann["bbox"], trans_output, height, width
            )
            if cls_id <= 0 or ("iscrowd" in ann and ann["iscrowd"] > 0):
                self._mask_ignore_or_crowd(ret, cls_id, bbox)
                continue
            vis_bbox.append(bbox)
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
        vis_bbox = [
            [
                bbox[0] / self.output_w * inp.shape[2],
                bbox[1] / self.output_h * inp.shape[1],
                bbox[2] / self.output_w * inp.shape[2],
                bbox[3] / self.output_h * inp.shape[1],
            ]
            for bbox in vis_bbox
        ]

        ret["c"] = c
        ret["s"] = np.asarray(s, dtype=np.float32)
        ret["image_id"] = img_id
        ret["output_size"] = np.asarray([self.output_h, self.output_w])
        ret["orig_size"] = np.asarray([height, width])
        ret["width"] = dataset_dict["width"]
        ret["height"] = dataset_dict["height"]
        # vis_image = ret["image"] / ret["image"].max() * 255
        # vis_image = vis_image.transpose(1, 2, 0)
        # vis_image = vis_image.astype(np.uint8)
        # self.vis_img(vis_image, img_id=20226, bbox_xyxy=vis_bbox, toxyxy=False, heatmap=None)
        assert self.is_train, "MOT17DatasetMapper should only be used for training!"
        img_id = dataset_dict["image_id"]
        d2_ori, d2_inp, d2_boxes, d2_anns = self.d2_load_data_from_img_id(img_id=img_id)
        num_objs = min(len(d2_anns), self.max_objs)
        d2_ret = self.d2_ret_init()
        d2_ret["image"], d2_ret["pad_mask"] = d2_inp, np.ones(d2_inp.shape[1:])
        if self.tracking:
            pre_img_id = img_info["prev_image_id"]
            if pre_img_id == -1 or pre_img_id not in self.images:
                pre_img_id = img_id
            else:
                pre_img_id = img_info["prev_image_id"]
            pre_d2_ori, pre_d2_inp, pre_d2_boxes, _ = self.d2_load_data_from_img_id(
                img_id=pre_img_id
            )
            d2_ret["pre_img"], d2_ret["pre_pad_mask"] = pre_d2_inp, np.ones(
                pre_d2_inp.shape[1:]
            )

        for k in range(num_objs):
            ann = d2_anns[k]
            bbox = d2_boxes[k]
            cls_id = int(self.cat_ids[ann["category_id"]])
            if cls_id > self.num_classes or cls_id <= -999:
                continue
            if cls_id <= 0 or ("iscrowd" in ann and ann["iscrowd"] > 0):
                self._mask_ignore_or_crowd(ret, cls_id, bbox)
                continue
            scaled_bbox = self.scale_bbox_to_output_space(
                bbox,
                d2_inp.shape[2],
                d2_inp.shape[1],
                self.output_w,
                self.output_h,
                clip=True,
            )
            d2_ret = self.d2_add_instance(
                d2_ret,
                k,
                cls_id,
                scaled_bbox,
            )
        d2_ret["width"] = d2_inp.shape[2]
        d2_ret["height"] = d2_inp.shape[1]
        d2_ret["output_size"] = np.array([self.output_h, self.output_w])
        # vis_image = d2_ret["image"] / d2_ret["image"].max() * 255
        # vis_image = vis_image.transpose(1, 2, 0)
        # vis_image = vis_image.astype(np.uint8)
        # vis_bbox = [
        #     [
        #         bbox[0] / self.output_w * inp.shape[2],
        #         bbox[1] / self.output_h * inp.shape[1],
        #         bbox[2] / self.output_w * inp.shape[2],
        #         bbox[3] / self.output_h * inp.shape[1],
        #     ]
        #     for bbox in d2_ret["boxes"]
        # ]
        # self.vis_img(
        #     vis_image, img_id=20227, bbox_xyxy=vis_bbox, toxyxy=False, heatmap=None
        # )
        return ret

    def d2_ret_init(self):
        max_objs = self.max_objs
        d2_ret = {}
        d2_ret["hm"] = np.zeros(
            (self.num_classes, self.output_h, self.output_w), np.float32
        )
        d2_ret["ind"] = np.zeros((max_objs), dtype=np.int64)
        d2_ret["cat"] = np.zeros((max_objs), dtype=np.int64)
        d2_ret["mask"] = np.zeros((max_objs), dtype=np.float32)
        d2_ret["boxes"] = np.zeros((max_objs, 4), dtype=np.float32)
        d2_ret["boxes_mask"] = np.zeros((max_objs), dtype=np.float32)
        d2_ret["center_offset"] = np.zeros((max_objs, 2), dtype=np.float32)
        regression_head_dims = {
            "reg": 2,
            "wh": 2,
            # "tracking": 2,
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
            if head in self.heads:
                d2_ret[head] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32
                )
                d2_ret[head + "_mask"] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32
                )

        if "hm_hp" in self.heads:
            raise RuntimeError("hm_hp has been deprecated!")

        if "rot" in self.heads:
            raise RuntimeError("rot has been deprecated!")

        return d2_ret

    def d2_load_data_from_img_id(self, img_id):
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info["file_name"]
        img_p = os.path.join(self.data_dir, file_name)
        image = utils.read_image(img_p)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(self.coco.loadAnns(ids=ann_ids))
        # essential step to pad the image to the target ratio
        h, w, c = image.shape
        target_ratio = 1.0 * self.input_w / self.input_h
        if 1.0 * w / h < target_ratio:
            new_w = int(target_ratio * h)
            new_img = np.zeros((h, new_w, c)).astype(image.dtype)
            new_img[:, :w, :] = image
            if "width" in img_info.keys():
                img_info["width"] = new_w
        else:
            new_img = image
        image = new_img
        utils.check_image_size(img_info, image)
        boxes = self.fetch_bboxes_from_annotations(anns)
        aug_input = T.AugInput(image, boxes=boxes)
        aug_input, _ = T.apply_transform_gens(self.augs, aug_input)
        aug_input = self.apply_resize_aug(aug_input)
        image = aug_input.image
        boxes = torch.tensor(aug_input.boxes)
        ori_image = image.copy()
        d2_inp = self.image_normalize(image)
        d2_inp = np.ascontiguousarray(d2_inp.transpose(2, 0, 1))
        return ori_image, d2_inp, boxes, anns

    def _load_data_given_img_id(self, img_id):
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(self.data_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(self.coco.loadAnns(ids=ann_ids))
        assert os.path.exists(img_path), f"{img_path} does not exist!"
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # commented out by Zelin
        # padding before affine warping to prevent cropping
        h, w, c = img.shape
        target_ratio = 1.0 * self.input_w / self.input_h
        if 1.0 * w / h < target_ratio:
            new_w = int(target_ratio * h)
            new_img = np.zeros((h, new_w, c)).astype(img.dtype)
            new_img[:, :w, :] = img
            if "width" in img_info.keys():
                img_info["width"] = new_w
        else:
            new_img = img
        return new_img, anns, img_info, img_path, np.ones_like(img)

    def _init_ret(self, ret, gt_det):
        max_objs = self.max_objs
        ret["hm"] = np.zeros(
            (self.num_classes, self.output_h, self.output_w), np.float32
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
            if head in self.heads:
                ret[head] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32
                )
                ret[head + "_mask"] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32
                )
                gt_det[head] = []

        if "hm_hp" in self.heads:
            num_joints = self.num_joints
            ret["hm_hp"] = np.zeros(
                (num_joints, self.output_h, self.output_w), dtype=np.float32
            )
            ret["hm_hp_mask"] = np.zeros((max_objs * num_joints), dtype=np.float32)
            ret["hp_offset"] = np.zeros((max_objs * num_joints, 2), dtype=np.float32)
            ret["hp_ind"] = np.zeros((max_objs * num_joints), dtype=np.int64)
            ret["hp_offset_mask"] = np.zeros(
                (max_objs * num_joints, 2), dtype=np.float32
            )
            ret["joint"] = np.zeros((max_objs * num_joints), dtype=np.int64)

        if "rot" in self.heads:
            ret["rotbin"] = np.zeros((max_objs, 2), dtype=np.int64)
            ret["rotres"] = np.zeros((max_objs, 2), dtype=np.float32)
            ret["rot_mask"] = np.zeros((max_objs), dtype=np.float32)
            gt_det.update({"rot": []})

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
        return bbox, bbox_amodal

    def _get_input(self, img, trans_input, padding_mask=None):
        img = img.copy()
        if padding_mask is None:
            padding_mask = np.ones_like(img)
        inp = cv2.warpAffine(
            img, trans_input, (self.input_w, self.input_h), flags=cv2.INTER_LINEAR
        )

        # to mask = 1 (padding part), not to mask = 0
        affine_padding_mask = cv2.warpAffine(
            padding_mask,
            trans_input,
            (self.input_w, self.input_h),
            flags=cv2.INTER_LINEAR,
        )
        affine_padding_mask = affine_padding_mask[:, :, 0]
        affine_padding_mask[affine_padding_mask > 0] = 1

        inp = inp.astype(np.float32) / 255.0
        if self.split == "train" and self.color_aug and np.random.rand() < 0.2:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp, 1 - affine_padding_mask

    def _flip_anns(self, anns, width):
        for k in range(len(anns)):
            bbox = anns[k]["bbox"]
            anns[k]["bbox"] = [width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

            if "hps" in self.heads and "keypoints" in anns[k]:
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

            if "rot" in self.heads and "alpha" in anns[k]:
                anns[k]["alpha"] = (
                    np.pi - anns[k]["alpha"]
                    if anns[k]["alpha"] > 0
                    else -np.pi - anns[k]["alpha"]
                )

            if "amodel_offset" in self.heads and "amodel_center" in anns[k]:
                anns[k]["amodel_center"][0] = width - anns[k]["amodel_center"][0] - 1
        return anns

    def _img_preprocess(self, anns, img, pad_img):
        img_blurred = False
        if self.image_blur_aug and np.random.rand() < 0.1 and self.split == "train":
            img = self.blur_aug(img)
            img_blurred = True

        # get image height and width
        height, width = img.shape[0], img.shape[1]
        # get image centers
        c = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0], dtype=np.float32)
        # get image size or max h or max w
        s = (
            max(img.shape[0], img.shape[1]) * 1.0
            if self.max_crop
            else np.array([img.shape[1], img.shape[0]], np.float32)
        )
        aug_s, rot, flipped = 1, 0, 0
        if self.split == "train":
            # drift image centers, change image size with scale, rotate image with rot.
            c, aug_s, rot = self._get_aug_param(c, s, width, height)
            s = s * aug_s
            # random flip
            if np.random.random() < self.flip_ratio:
                flipped = 1
                img = img[:, ::-1, :].copy()
                anns = self._flip_anns(anns, width)

        # we will reshape image to standard input shape, trans_input =transform for resizing gt to input size
        trans_input = get_affine_transform(c, s, rot, [self.input_w, self.input_h])
        # the output heatmap size != input size, trans_output = transform for resizing gt to output size
        trans_output = get_affine_transform(c, s, rot, [self.output_w, self.output_h])
        inp, padding_mask = self._get_input(img, trans_input, padding_mask=pad_img)
        return (
            inp,
            padding_mask,
            img_blurred,
            flipped,
            trans_input,
            trans_output,
            c,
            aug_s,
            s,
            rot,
            width,
            height,
        )

    def _coco_box_to_bbox(self, box):
        bbox = np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32
        )
        return bbox

    def _get_pre_dets(self, anns, trans_input):
        hm_h, hm_w = self.input_h, self.input_w
        down_ratio = self.down_ratio
        trans = trans_input
        reutrn_hm = self.pre_hm
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
        pre_cts, track_ids = [], []
        for ann in anns:
            cls_id = int(self.cat_ids[ann["category_id"]])
            if (
                cls_id > self.num_classes
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
                ct[0] = ct[0] + np.random.randn() * self.hm_disturb * w
                ct[1] = ct[1] + np.random.randn() * self.hm_disturb * h
                conf = 1 if np.random.random() > self.lost_disturb else 0

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
                if np.random.random() < self.fp_disturb and reutrn_hm:
                    ct2 = ct0.copy()
                    # Hard code heatmap disturb ratio, haven't tried other numbers.
                    ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                    ct2[1] = ct2[1] + np.random.randn() * 0.05 * h
                    ct2_int = ct2.astype(np.int32)
                    draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)
        return pre_hm, pre_cts, track_ids

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
        radius = gaussian_radius((math.ceil(h_clip), math.ceil(w_clip)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
        )
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
        ret["ind"][k] = ct_int[1] * self.output_w + ct_int[0]
        # the .xxx part of the kpts
        ret["reg"][k] = ct - ct_int
        ret["reg_mask"][k] = 1

        # center_offset
        ret["center_offset"][k] = (
            0.5 * (bbox_amodal[0] + bbox_amodal[2]) - ct[0],
            0.5 * (bbox_amodal[1] + bbox_amodal[3]) - ct[1],
        )

        ret["center_offset_mask"][k] = 1
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
        ret["boxes"][k][0::2] /= self.output_w
        ret["boxes"][k][1::2] /= self.output_h
        ret["boxes_mask"][k] = 1
        gt_det["scores"].append(1)
        gt_det["clses"].append(cls_id - 1)
        gt_det["cts"].append(ct)

        if "tracking" in self.heads:
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

    def d2_add_instance(
        self,
        d2_ret,
        k,
        cls_id,
        bbox,
    ):
        # Note: box must be in the output image plane, add it to gt heatmap
        bbox_height, bbox_width = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if bbox_height <= 0 or bbox_width <= 0:
            return d2_ret
        radius = gaussian_radius((math.ceil(bbox_height), math.ceil(bbox_width)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
        )
        ct_int = ct.astype(np.int32)
        # 'cat': categories of shape [num_objects], recording the cat id.
        d2_ret["cat"][k] = cls_id - 1
        # 'mask': mask of shape [num_objects], if mask == 1, to train, if mask == 0, not to train.
        d2_ret["mask"][k] = 1
        if "wh" in d2_ret:
            # 'wh' = box_amodal size,of shape [num_objects, 2]
            d2_ret["wh"][k] = 1.0 * bbox_width, 1.0 * bbox_height
            d2_ret["wh_mask"][k] = 1
        # 'ind' of shape [num_objects],
        # indicating the position of the object = y*W_output + x in a heatmap of shape [out_h, out_w]
        d2_ret["ind"][k] = ct_int[1] * self.output_w + ct_int[0]
        assert d2_ret["ind"][k] < self.output_w * self.output_h
        # the .xxx part of the kpts
        d2_ret["reg"][k] = ct - ct_int
        d2_ret["reg_mask"][k] = 1
        # center_offset
        d2_ret["center_offset"][k] = (
            0.5 * (bbox[0] + bbox[2]) - ct[0],
            0.5 * (bbox[1] + bbox[3]) - ct[1],
        )
        d2_ret["center_offset_mask"][k] = 1
        d2_ret["hm"][cls_id - 1] = draw_umich_gaussian(
            d2_ret["hm"][cls_id - 1], ct_int, radius
        )
        d2_ret["boxes_mask"][k] = 1
        d2_ret["boxes"][k] = np.asarray(
            [
                0.5 * (bbox[0] + bbox[2]),
                0.5 * (bbox[1] + bbox[3]),
                (bbox[2] - bbox[0]),
                (bbox[3] - bbox[1]),
            ],
            dtype=np.float32,
        )
        d2_ret["boxes"][k][0::2] /= self.output_w
        d2_ret["boxes"][k][1::2] /= self.output_h
        return d2_ret

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

    ## data augmentations
    def _get_aug_param(self, c, s, width, height, disturb=False):
        if (not self.rand_crop) and not disturb:
            sf = self.sf
            cf = self.cf

            if type(s) == float or type(s) == np.float64 or type(s) == np.float32:
                s = [s, s]
            c[0] += s[0] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s[1] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        else:
            sf = self.rand_crop_scale
            cf = self.rand_crop_shift
            # print(s)
            if type(s) == float or type(s) == np.float64 or type(s) == np.float32:
                s = [s, s]
            c[0] += s[0] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s[1] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        rot = 0  # forbid rotate augmentation

        return c, aug_s, rot
