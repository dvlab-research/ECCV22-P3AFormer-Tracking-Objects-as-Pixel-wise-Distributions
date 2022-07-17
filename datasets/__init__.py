# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .coco import build as build_coco
from .detmot import build as build_e2e_mot
from .static_detmot import build as build_e2e_static_mot
from .joint import build as build_e2e_joint
from .torchvision_datasets import CocoDetection
from .byte_mot_half import build as build_byte_mot_val
from .p3aformer_dataset.mot17_train import build as build_p3aformer_mot
from .p3aformer_dataset.crowdhuman import CrowdHuman, build_crowdhuman
from .p3aformer_dataset.mot17_train import build as build_p3aformer_mot_mixed


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == "coco":
        return build_coco(image_set, args)
    if args.dataset_file == "coco_panoptic":
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic

        return build_coco_panoptic(image_set, args)
    if args.dataset_file == "e2e_joint":  # default dataset for MOT task
        return build_e2e_joint(image_set, args)
    if args.dataset_file == "e2e_static_mot":
        return build_e2e_static_mot(image_set, args)
    if args.dataset_file == "e2e_mot":
        return build_e2e_mot(image_set, args)
    if args.dataset_file == "byte_mot_half":
        return build_byte_mot_val(image_set, args)
    if args.dataset_file == "p3aformer_mot":
        return build_p3aformer_mot(image_set, args)
    if args.dataset_file == "p3aformer_mixed":
        return build_p3aformer_mot_mixed(image_set, args)
    if args.dataset_file == "crowdHuman":
        return build_crowdhuman(image_set, args)
    raise ValueError(f"dataset {args.dataset_file} not supported")
