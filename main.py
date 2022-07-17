import os

os.environ["OMP_NUM_THREADS"] = "4"
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
from util.tool import load_model
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import (
    motr_evaluate,
    p3aformer_evaluate,
    motr_train_one_epoch,
    train_one_epoch_mot,
    p3aformer_train_one_epoch,
)
from models import build_model
from datasets.p3aformer_dataset.crowdhuman import CrowdHuman
import pdb
import os

# xyh #
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = False

os.environ["OMP_NUM_THREADS"] = "4"


def get_args_parser():
    parser = argparse.ArgumentParser("Deformable DETR Detector", add_help=False)
    parser.add_argument(
        "--dataset_name",
        required=True,
        type=str,
        choices=("MOT15", "MOT17"),
        help="dataset name",
    )
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone.0"], type=str, nargs="+"
    )
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--save_period", default=50, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument("--meta_arch", default="deformable_detr", type=str)
    parser.add_argument("--half_train", action="store_true")
    parser.add_argument("--sgd", action="store_true")
    parser.add_argument("--submit", action="store_true")

    # Variants of Deformable DETR
    parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--two_stage", default=False, action="store_true")
    parser.add_argument("--accurate_ratio", default=False, action="store_true")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    parser.add_argument("--num_anchors", default=1, type=int)

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument("--enable_fpn", action="store_true")
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--position_embedding_scale",
        default=2 * np.pi,
        type=float,
        help="position / size * scale",
    )
    parser.add_argument(
        "--num_feature_levels", default=4, type=int, help="number of feature levels"
    )
    parser.add_argument(
        "--learnable_queries",
        action="store_true",
        help="If true, we use learnable parameters.",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=300, type=int, help="Number of query slots"
    )
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)
    parser.add_argument("--decoder_cross_self", default=False, action="store_true")
    parser.add_argument("--sigmoid_attn", default=False, action="store_true")
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--cj", action="store_true")
    parser.add_argument("--extra_track_attn", action="store_true")
    parser.add_argument("--loss_normalizer", action="store_true")
    parser.add_argument("--max_size", default=1333, type=int)
    parser.add_argument("--val_width", default=800, type=int)
    parser.add_argument("--filter_ignore", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # * Matcher
    parser.add_argument(
        "--mix_match",
        action="store_true",
    )
    parser.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--offset_loss_coef", default=6, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # dataset parameters
    parser.add_argument(
        "--dataset_file",
        default="coco",
        choices=["coco", "crowdHuman", "p3aformer_mot"],
    )
    parser.add_argument("--gt_file_train", type=str)
    parser.add_argument("--gt_file_val", type=str)
    parser.add_argument(
        "--coco_path", default="/data/workspace/detectron2/datasets/coco/", type=str
    )
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--pretrained", default=None, help="resume from checkpoint")
    parser.add_argument(
        "--cache_mode",
        default=False,
        action="store_true",
        help="whether to cache images on memory",
    )

    # end-to-end mot settings.
    parser.add_argument("--mot_path", default="/data/Dataset/mot", type=str)
    parser.add_argument("--input_video", default="figs/demo.mp4", type=str)
    parser.add_argument(
        "--data_dir",
        default="./datasets/mot_mix",
        type=str,
        help="path to dataset txt split",
    )
    parser.add_argument(
        "--data_txt_path_train",
        default="./datasets/data_path/detmot17.train",
        type=str,
        help="path to dataset txt split",
    )
    parser.add_argument(
        "--data_txt_path_val",
        default="./datasets/data_path/detmot17.train",
        type=str,
        help="path to dataset txt split",
    )
    parser.add_argument("--img_path", default="data/valid/JPEGImages/")

    parser.add_argument("--query_interaction_layer", default="QIM", type=str, help="")
    parser.add_argument("--sample_mode", type=str, default="fixed_interval")
    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument("--random_drop", type=float, default=0)
    parser.add_argument("--fp_ratio", type=float, default=0)
    parser.add_argument("--merger_dropout", type=float, default=0.1)
    parser.add_argument("--update_query_pos", action="store_true")

    parser.add_argument("--sampler_steps", type=int, nargs="*")
    parser.add_argument("--sampler_lengths", type=int, nargs="*")
    parser.add_argument("--exp_name", default="submit", type=str)
    parser.add_argument("--memory_bank_score_thresh", type=float, default=0.0)
    parser.add_argument("--memory_bank_len", type=int, default=4)
    parser.add_argument("--memory_bank_type", type=str, default=None)
    parser.add_argument(
        "--memory_bank_with_self_attn", action="store_true", default=False
    )
    # training
    parser.add_argument(
        "--heads",
        default=["hm", "reg", "wh", "center_offset", "tracking"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--hm_weight", default=1, type=float)
    parser.add_argument("--off_weight", default=1, type=float)
    parser.add_argument("--wh_weight", default=0.1, type=float)
    parser.add_argument("--tracking_weight", default=1, type=float)
    parser.add_argument("--ct_offset_weight", default=0.1, type=float)
    parser.add_argument("--boxes_weight", default=0.5, type=float)
    parser.add_argument("--giou_weight", default=0.4, type=float)
    parser.add_argument("--norm_factor", default=1.0, type=float)
    parser.add_argument("--coco_panargsic_path", type=str)

    # centers
    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument("--input_h", default=640, type=int)
    parser.add_argument("--input_w", default=1088, type=int)
    parser.add_argument("--down_ratio", default=4, type=int)
    parser.add_argument("--dense_reg", type=int, default=1, help="")
    parser.add_argument(
        "--trainval",
        action="store_true",
        help="include validation in training and test on test set",
    )
    parser.add_argument(
        "--K", type=int, default=300, help="max number of output objects."
    )
    parser.add_argument("--debug", action="store_true")

    # noise
    parser.add_argument(
        "--not_rand_crop",
        action="store_true",
        help="not use the random crop data augmentationfrom CornerNet.",
    )
    parser.add_argument(
        "--not_max_crop",
        action="store_true",
        help="used when the training dataset has inbalanced aspect ratios.",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=0.0,
        help="when not using random crop apply shift augmentation.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="when not using random crop apply scale augmentation.",
    )
    parser.add_argument(
        "--rotate",
        type=float,
        default=0,
        help="when not using random crop apply rotation augmentation.",
    )
    parser.add_argument(
        "--flip",
        type=float,
        default=0.0,
        help="probability of applying flip augmentation.",
    )
    parser.add_argument(
        "--no_color_aug",
        action="store_true",
        help="not use the color augmenation from CornerNet",
    )
    parser.add_argument(
        "--image_blur_aug", action="store_true", help="blur image for aug."
    )
    parser.add_argument(
        "--aug_rot",
        type=float,
        default=0,
        help="probability of applying rotation augmentation.",
    )

    # tracking
    parser.add_argument(
        "--detr_path", default="/data/Dataset/mot", type=str
    )
    parser.add_argument(
        "--reid_path", default="/data/Dataset/mot", type=str
    )
    parser.add_argument("--max_frame_dist", type=int, default=3)
    parser.add_argument("--merge_mode", type=int, default=1)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--pre_hm", action="store_true")
    parser.add_argument("--same_aug_pre", action="store_true")
    parser.add_argument("--zero_pre_hm", action="store_true")
    parser.add_argument("--hm_disturb", type=float, default=0.05)
    parser.add_argument("--lost_disturb", type=float, default=0.4)
    parser.add_argument("--fp_disturb", type=float, default=0.1)
    parser.add_argument("--pre_thresh", type=float, default=-1)
    parser.add_argument("--track_thresh", type=float, default=0.3)
    parser.add_argument("--new_thresh", type=float, default=0.3)
    parser.add_argument("--ltrb_amodal", action="store_true")
    parser.add_argument("--ltrb_amodal_weight", type=float, default=0.1)
    parser.add_argument("--public_det", action="store_true")
    parser.add_argument("--no_pre_img", action="store_true")
    parser.add_argument("--zero_tracking", action="store_true")
    parser.add_argument("--hungarian", action="store_true")
    parser.add_argument("--max_age", type=int, default=-1)
    parser.add_argument("--out_thresh", type=float, default=-1, help="")
    parser.add_argument("--adaptive_clip", action="store_true", help="adaptive_clip")

    """ 
    ByteTrack settings
    """

    parser.add_argument("--no_byte", action="store_true", help="forbid byte track")
    parser.add_argument(
        "--track_thre", type=float, default=0.8, help="threshold to form a track"
    )
    parser.add_argument(
        "--low_thre", type=float, default=0.2, help="low threshold for bytetracker"
    )
    parser.add_argument(
        "--first_assign_thre",
        type=float,
        default=0.8,
        help="first assign threshold for bytetracker",
    )
    parser.add_argument(
        "--second_assign_thre",
        type=float,
        default=0.5,
        help="second assign threshold for bytetracker",
    )
    return parser


def main(args):
    utils.init_distributed_mode(args)
    args.eval = False
    my_map = 0
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(f"Start training P3AFormer, exp id: {args.exp_name}.")
    device = torch.device(args.device)

    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    if args.dataset_file in [
        "e2e_mot",
        "mot",
        "ori_mot",
        "e2e_static_mot",
        "e2e_joint",
    ]:
        collate_fn = utils.mot_collate_fn
    elif (
        args.dataset_file == "p3aformer_mot"
        or args.dataset_file == "p3aformer_mixed"
        or args.dataset_file == "crowdHuman"
    ):
        collate_fn = None
    else:
        collate_fn = utils.collate_fn
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_bs = 5 if args.meta_arch == "p3aformer" else args.batch_size
    data_loader_val = DataLoader(
        dataset_val,
        val_bs,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    if args.meta_arch == "p3aformer":
        for n, p in model_without_ddp.named_parameters():
            if match_name_keywords(n, args.lr_backbone_names):
                p.requires_grad = False

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif args.meta_arch == "p3aformer" or args.meta_arch == "transcenter":
        base_ds = dataset_val.coco
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
    assert base_ds is not None, "the detection gt is not successfully loaded!"

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    if args.pretrained is not None:
        model_without_ddp = load_model(model_without_ddp, args.pretrained)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint["model"], strict=False
        )
        unexpected_keys = [
            k
            for k in unexpected_keys
            if not (k.endswith("total_params") or k.endswith("total_ops"))
        ]
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))

        # check the resumed model
        if not args.eval:
            evaluate = (
                motr_evaluate
                if args.meta_arch != "p3aformer" and args.meta_arch != "transcenter"
                else p3aformer_evaluate
            )
            test_stats, coco_evaluator = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
            )

            # valbest save#
            avg_map = np.mean(
                [
                    test_stats["coco_eval_bbox"][0],
                    test_stats["coco_eval_bbox"][1],
                    test_stats["coco_eval_bbox"][3],
                    test_stats["coco_eval_bbox"][4],
                    test_stats["coco_eval_bbox"][5],
                ]
            )
            if avg_map >= my_map and args.start_epoch > 1:
                my_map = float(avg_map)
                print("resume map ", my_map)
            else:
                print("first epoch, init map:", my_map)

    if args.eval:
        if args.meta_arch == "p3aformer":
            torch.cuda.empty_cache()
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
        )
        if args.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
            )
        return

    print("Start training")
    start_time = time.time()

    if args.dataset_file in [
        "e2e_mot",
        "mot",
        "ori_mot",
        "e2e_static_mot",
        "e2e_joint",
    ]:
        train_func = train_one_epoch_mot
        dataset_train.set_epoch(args.start_epoch)
        dataset_val.set_epoch(args.start_epoch)
    elif (
        args.dataset_file == "p3aformer_mot"
        or args.dataset_file == "p3aformer_mixed"
        or args.dataset_file == "crowdHuman"
    ):
        train_func = p3aformer_train_one_epoch
    else:
        train_func = motr_train_one_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_func(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
        )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = []
            # extra checkpoint before LR drop and every 5 epochs
            if (
                (epoch + 1) % args.lr_drop == 0
                or (epoch + 1) % args.save_period == 0
                or (
                    ((args.epochs >= 100 and (epoch + 1) > 100) or args.epochs < 100)
                    and (epoch + 1) % 5 == 0
                )
            ):
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )
        if (
            args.dataset_file
            not in ["e2e_mot", "mot", "ori_mot", "e2e_static_mot", "e2e_joint"]
            and (epoch + 1) % 5 == 0
        ):
            evaluate = p3aformer_evaluate
            test_stats, coco_evaluator = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
            )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if args.dataset_file in [
            "e2e_mot",
            "mot",
            "ori_mot",
            "e2e_static_mot",
            "e2e_joint",
        ]:
            dataset_train.step_epoch()
            dataset_val.step_epoch()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Deformable DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
