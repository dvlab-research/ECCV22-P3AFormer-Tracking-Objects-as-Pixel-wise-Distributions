# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from __future__ import print_function
import os
import numpy as np
import argparse
import torchvision.transforms.functional as F
import torch
import csv
import os.path as osp
from torch.utils.data import DataLoader
from pathlib import Path
from models import build_model
from tracker.dense_tracker.dense_tracker import Tracker
from datasets.p3aformer_dataset.mot17_val_save_mem import MOT17_val
from datasets.p3aformer_dataset.mot15_val_save_mem import MOT15_val
from tracker.common.track_structure_transfer import frame_first_to_id_first
from main import get_args_parser
from util.evaluation import Evaluator
import motmetrics as mm
import yaml
import pdb
from shutil import copyfile
from util.image import get_affine_transform
from tools.visualization_tool import Visualizer
from util.tool import load_model
from util.system import remove_files_under_folder
from tracker.byte_tracker.byte_tracker import BYTETracker


np.random.seed(2022)


def write_results(all_tracks, out_dir, seq_name=None, frame_offset=0):
    output_dir = out_dir + "/txt/"
    """Write the tracks in the format for MOT16/MOT17 submission
    all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num  if frame_first=False,
    Each file contains these lines:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"
    assert seq_name is not None, "[!] No seq_name, probably using combined database"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file = osp.join(output_dir, seq_name + ".txt")
    with open(file, "w") as of:
        writer = csv.writer(of, delimiter=",")
        for i in sorted(all_tracks):
            track = all_tracks[i]
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                writer.writerow(
                    [
                        frame + frame_offset,
                        i + 1,
                        x1 + 1,
                        y1 + 1,
                        x2 - x1 + 1,
                        y2 - y1 + 1,
                        -1,
                        -1,
                        -1,
                        -1,
                    ]
                )
    # TODO: validate this in MOT15
    # copy to FRCNN, DPM.txt, private setting
    copyfile(file, file[:-7] + "FRCNN.txt")
    copyfile(file, file[:-7] + "DPM.txt")
    return file


if __name__ == "__main__":
    # handle configs
    parser = argparse.ArgumentParser("Eval p3aformer", parents=[get_args_parser()])
    args = parser.parse_args()
    args.eval = True
    is_val = not args.submit
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Removing all existing files in the output directory: {args.output_dir}")
        remove_files_under_folder(args.output_dir, select_str="txt")
    use_byte = not args.no_byte
    dataset_name = args.dataset_name
    print(f"Using Byte Track Augmentation: {use_byte}.")
    center_pred = args.meta_arch == "transcenter" or args.meta_arch == "p3aformer"
    if center_pred:
        with open("configs/detracker_reidV3.yaml", "r") as f:
            tracktor = yaml.safe_load(f)["tracktor"]
        with open("configs/reid.yaml", "r") as f:
            reid = yaml.safe_load(f)["reid"]
        args.input_h, args.input_w = 640, 1088
        args.output_h = args.input_h // args.down_ratio
        args.output_w = args.input_w // args.down_ratio
        args.input_res = max(args.input_h, args.input_w)
        args.output_res = max(args.output_h, args.output_w)
        args.track_thresh = tracktor["tracker"]["track_thresh"]
        args.pre_thresh = tracktor["tracker"]["pre_thresh"]
        args.new_thresh = max(
            tracktor["tracker"]["track_thresh"], tracktor["tracker"]["new_thresh"]
        )
        args.node0 = True
        args.private = True

    # build visualizer
    vis = Visualizer() if args.vis else None

    # build models, load models, and send to cuda
    detr, criterion, postprocessors = build_model(args)
    if center_pred:
        detr.load_state_dict(torch.load(args.detr_path)["model"])
        detr.cuda().eval()
        tracker = Tracker(
            detr, tracktor["tracker"], postprocessor=postprocessors["bbox"]
        )
        tracker.public_detections = False
    else:
        detr = load_model(detr, args.resume)
        detr = detr.cuda()
        detr.eval()

    # build datasets
    if center_pred:
        if dataset_name == "MOT15":
            ds = MOT15_val(
                args, "train"
            )  # using MOT15 training split to test MOT17 models.
        elif dataset_name == "MOT17" and not is_val:
            ds = MOT17_val(args, "test")
        elif dataset_name == "MOT17":
            ds = MOT17_val(args, "train")
            # using train as eval, the half validation does not work here for P3AFormer
        else:
            raise NotImplementedError(f"Not implemented dataset {dataset_name}.")
        using_mot17 = ds.is_mot17
        shuffle = vis is not None
        data_loader = DataLoader(
            ds, 1, shuffle=shuffle, drop_last=False, num_workers=0, pin_memory=True
        )
        output_dir = args.output_dir
        all_accs, all_seqs = [], []
        for seq, seq_n in data_loader:
            seq_name = seq_n[0]
            seq_num = "SDP" if (using_mot17 and not is_val) else seq_name
            if (seq_num not in seq_name) and using_mot17 and not is_val:
                del seq
                continue
            print("Inference on seq_name: ", seq_name)
            tracker.reset()
            keys = list(seq.keys())
            keys.pop(keys.index("v_id"))
            frames_list = sorted(keys)
            frame_offset = 0
            v_id = seq["v_id"].item()
            pub_dets = ds.VidPubDet[v_id]
            c = None
            s = None
            trans_input = None
            bt = (
                BYTETracker(
                    track_thre=args.track_thre,
                    low_thre=args.low_thre,
                    first_assign_thre=args.first_assign_thre,
                    second_assign_thre=args.second_assign_thre,
                )
                if use_byte
                else None
            )
            bt_results = {}
            for idx, frame_name in enumerate(frames_list):
                blob = seq[frame_name]
                frame_id = blob["frame_id"].item()
                img_id = blob["img_id"].item()
                pub_det = pub_dets[frame_id - 1]
                img, _, img_info, _, pad_mask = ds._load_data(img_id)
                if vis:
                    vis.add_img(img, img_id=idx)
                height, width = img.shape[0], img.shape[1]
                if c is None:
                    # get image centers
                    c = np.array(
                        [img.shape[1] / 2.0, img.shape[0] / 2.0], dtype=np.float32
                    )
                    # get image size or max h or max w
                    s = (
                        max(img.shape[0], img.shape[1]) * 1.0
                        if not ds.opt.not_max_crop
                        else np.array([img.shape[1], img.shape[0]], np.float32)
                    )

                aug_s, rot, flipped = 1, 0, 0
                if trans_input is None:
                    # resize input
                    trans_input = get_affine_transform(
                        c, s, rot, [ds.opt.input_w, ds.opt.input_h]
                    )
                inp, padding_mask = ds._get_input(
                    img, trans_input, padding_mask=pad_mask
                )

                # load a pre image with random interval #  # TODO: validate this comment
                pre_image, _, frame_dist, pre_img_id, pre_pad_mask = ds._load_pre_data(
                    img_info["video_id"], img_info["frame_id"]
                )
                pre_inp, pre_padding_mask = ds._get_input(
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
                if idx == 0:
                    frame_offset = int(frame_name[:-4])
                    print("frame offset : ", frame_offset)
                print(
                    f"step frame: {int(frame_name[:-4])} / {len(frames_list)}.",
                    end="\r",
                )
                batch["frame_name"] = frame_name
                batch["video_name"] = seq_name
                det, score = tracker.step(batch)

                if not seq_num in seq_name and using_mot17:
                    continue
                else:
                    if use_byte:
                        cur_results = torch.cat([det, score.view(-1, 1)], dim=1)
                        online_targets = bt.update(cur_results.cpu().numpy())
                        online_ret = []
                        for t in online_targets:
                            online_ret.append(
                                [
                                    t.tlbr[0],
                                    t.tlbr[1],
                                    t.tlbr[2],
                                    t.tlbr[3],
                                    t.score,
                                    t.track_id,
                                ]
                            )
                        bt_results[idx] = online_ret
                        tracker.results = frame_first_to_id_first(bt_results)
                    if vis:
                        results = tracker.get_results(frame_first=False)
                        for track_id in results:
                            if idx not in results[track_id]:
                                continue
                            cur_track_res = results[track_id][idx]
                            cur_conf = cur_track_res[4]
                            if cur_conf >= args.track_thre:
                                ass_rank = "first"
                            elif cur_conf > args.low_thre:
                                ass_rank = "second"
                            else:
                                ass_rank = "none"
                            vis.add_coco_bbox(
                                cur_track_res[:4],
                                0,
                                conf=track_id,
                                add_txt="_" + ass_rank,
                                img_id=idx,
                            )
                            # before_ct = (int((cur_track_res[:4][0] + cur_track_res[:4][2]) / 2), int((cur_track_res[:4][1] + cur_track_res[:4][3]) / 2))
                            # after_ct = (int(cur_track_res[-2]), int(cur_track_res[-1]))
                            # diff_ct = (after_ct[0] - before_ct[0], after_ct[1] - before_ct[1])
                            # vis.add_arrow(before_ct, diff_ct, img_id=idx)
                        vis.save_video(path=args.output_dir)
                        print(
                            "Visualization video is saved at: ",
                            args.output_dir,
                            end="\r",
                        )
                    results = tracker.get_results(frame_first=False)
                    save_path = write_results(
                        results,
                        args.output_dir,
                        seq_name=seq_name,
                        frame_offset=frame_offset,
                    )
                    print("Write txt results at: ", save_path, end="\r")

            if args.dataset_name == "MOT15" or is_val:
                # train_dir = os.path.join(args.data_dir, 'images/train')
                train_dir = os.path.join(args.data_dir, "train")
                evaluator = Evaluator(train_dir, seq_num)
                accs = evaluator.eval_file(save_path)
                all_accs.append(accs)
                all_seqs.append(seq_num)
                metrics = mm.metrics.motchallenge_metrics
                mh = mm.metrics.create()
                summary = Evaluator.get_summary(all_accs, all_seqs, metrics)
                strsummary = mm.io.render_summary(
                    summary,
                    formatters=mh.formatters,
                    namemap=mm.io.motchallenge_metric_names,
                )
                print(strsummary)
    else:
        seq_nums = [
            "ADL-Rundle-6",
            "ETH-Bahnhof",
            "KITTI-13",
            "PETS09-S2L1",
            "TUD-Stadtmitte",
            "ADL-Rundle-8",
            "KITTI-17",
            "ETH-Pedcross2",
            "ETH-Sunnyday",
            "TUD-Campus",
            "Venice-2",
        ]
        accs = []
        seqs = []

        for seq_num in seq_nums:
            print("solve {}".format(seq_num))
            det = Detector(args, use_byte, model=detr, seq_num=seq_num)
            det.detect(vis=False)
            accs.append(det.eval_seq())
            seqs.append(seq_num)

            metrics = mm.metrics.motchallenge_metrics
            mh = mm.metrics.create()
            summary = Evaluator.get_summary(accs, seqs, metrics)
            strsummary = mm.io.render_summary(
                summary,
                formatters=mh.formatters,
                namemap=mm.io.motchallenge_metric_names,
            )
            print(strsummary)
            with open("eval_log.txt", "a") as f:
                print(strsummary, file=f)
