from collections import deque
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from util.p3aformer.tracker_util import bbox_overlaps, warp_pos
import math
from util.p3aformer.p3aformer_misc import (
    NestedTensor,
    gaussian_radius,
    affine_transform,
    draw_umich_gaussian,
)
import lap
from tracker.common.track_structure_transfer import *
from models.d2_p3aformer.d2_postprocess import PostProcess
from detectron2.config import configurable
from tracker.byte_tracker.byte_tracker import BYTETracker
from tools.visualization_tool import Visualizer
from tracker.d2_p3aformer.write_results import write_results


class P3AFormerTracker(nn.Module):
    cl = 1

    @configurable
    def __init__(
        self,
        p3aformer_model,
        visualizer: Visualizer = None,
        postprocessor=None,
        byte_association: BYTETracker = None,
        output_dir=None,
    ):
        super(P3AFormerTracker, self).__init__()
        self.p3aformer_model = p3aformer_model
        self.inactive_patience = 60  # deprecating
        self.max_features_num = 10  # deprecating
        self.postprocessor = postprocessor
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
        self.bt_results = {}
        self.img_features = None
        self.encoder_pos_encoding = None
        self.transforms = transforms.ToTensor()
        self.last_image = None
        self.pre_sample = None
        self.sample = None
        self.pre_img_features = None
        self.pre_encoder_pos_encoding = None
        self.visualizer = visualizer
        self.byte_association = byte_association
        self.output_dir = output_dir

    @classmethod
    def from_config(self, cfg, p3aformer_model):
        output_dir = cfg.OUTPUT_DIR
        visualizer = Visualizer() if cfg.TRACK.VIS else None
        bt = BYTETracker(
            track_thre=cfg.TRACK.DENSETRACK.TRACK_THRE,
            low_thre=cfg.TRACK.DENSETRACK.LOW_THRE,
            first_assign_thre=cfg.TRACK.DENSETRACK.FIRST_ASSIGN_THRE,
            second_assign_thre=cfg.TRACK.DENSETRACK.SECOND_ASSIGN_THRE,
        )
        post_processor = PostProcess(cfg)
        return {
            "p3aformer_model": p3aformer_model,
            "postprocessor": post_processor,
            "visualizer": visualizer,
            "byte_association": bt,
            "output_dir": output_dir,
        }

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []
        self.last_image = None
        self.pre_sample = None
        self.sample = None
        self.pre_img_features = None
        self.pre_encoder_pos_encoding = None
        self.flow = None
        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0
        self.frame_offset = 0
        self.bt_results = {}

    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])),
            )
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_a, unmatched_b

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features, pred_ct):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(
                Track(
                    new_det_pos[i].view(1, -1),
                    new_det_scores[i],
                    pred_ct[i],
                    self.track_num + i,
                    new_det_features[i].view(1, -1),
                    self.inactive_patience,
                    self.max_features_num,
                    1,
                )
            )
        self.track_num += num_new

    def detect_tracking(self, batch):
        hm_h, hm_w = self.pre_sample.tensors.shape[2], self.pre_sample.tensors.shape[3]
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32)
        trans = batch["trans_input"][0].cpu().numpy()
        # draw pre_hm with self.pos # pre track
        for bbox in self.get_pos().cpu().numpy():
            bbox[:2] = affine_transform(bbox[:2], trans)
            bbox[2:] = affine_transform(bbox[2:], trans)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            # draw gt heatmap with
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
                )
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(pre_hm[0], ct_int, radius, k=1)

        pre_hm = torch.from_numpy(pre_hm).cuda().unsqueeze_(0)
        outputs = self.p3aformer_model.get_detection_output(
            samples=self.sample,
            pre_samples=self.pre_sample,
            features=self.img_features,
            pos=self.encoder_pos_encoding,
            pre_features=self.pre_img_features,
            pre_pos=self.pre_encoder_pos_encoding,
            pre_hm=pre_hm,
        )
        results = self.postprocessor(outputs, batch["orig_size"], filter_score=True)[0]
        out_scores, labels_out, out_boxes, heatmap = (
            results["scores"],
            results["labels"],
            results["boxes"],
            results["heatmap"],
        )
        filtered_idx = labels_out == 1
        return (
            out_boxes[filtered_idx, :].cuda(),
            out_scores[filtered_idx].cuda(),
            heatmap,
        )

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos.cuda()
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], dim=0).cuda()
        else:
            pos = torch.zeros(0, 4).cuda()
        return pos

    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))

    @torch.no_grad()
    def step(self, image_idx, seq_name, frame_name, blob):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        if image_idx == 0:
            self.frame_offset = int(frame_name[:-4])
        # get backbone features
        self.sample = NestedTensor(blob["image"].cuda(), blob["pad_mask"].cuda())
        self.img_features, self.encoder_pos_encoding = self.p3aformer_model.backbone(
            self.sample
        )
        if self.pre_img_features is None:
            self.pre_sample = NestedTensor(
                blob["pre_image"].cuda(), blob["pre_pad_mask"].cuda()
            )
            (
                self.pre_img_features,
                self.pre_encoder_pos_encoding,
            ) = self.p3aformer_model.backbone(self.pre_sample)
        blob["img"] = (
            blob["orig_img"].clone().float() / 255.0
        )  # todo: consider deprecating this
        # detect
        raw_private_det_pos, raw_private_det_scores, heatmap = self.detect_tracking(
            blob
        )
        cur_results = torch.cat(
            [raw_private_det_pos, raw_private_det_scores.view(-1, 1)], dim=1
        )
        online_targets = self.byte_association.update(cur_results.cpu().numpy())
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
        self.bt_results[image_idx] = online_ret
        self.results = frame_first_to_id_first(self.bt_results)
        if self.visualizer:
            if image_idx > 20:
                pdb.set_trace()
            results = self.get_results(frame_first=False)
            for track_id in results:
                if image_idx not in results[track_id]:
                    continue
                cur_track_res = results[track_id][image_idx]
                self.visualizer.add_coco_bbox(
                    cur_track_res[:4],
                    0,
                    conf=track_id,
                    add_txt="_" + str(cur_track_res[4])[:4],  # confidence
                    img_id=image_idx,
                )
                # self.visualizer.add_heatmap(heatmap, image_idx)
            self.visualizer.save_video(path=self.output_dir, name=seq_name)
            print(
                "Visualization video is saved at: ",
                self.output_dir,
                end="\r",
            )
        results = self.get_results(frame_first=False)
        save_path = write_results(
            results,
            self.output_dir,
            seq_name=seq_name,
            frame_offset=self.frame_offset,
            verbose=False,
        )
        return save_path

    def get_results(self, frame_first):
        # get results, if frame first, then return a list of results else return a list of trackers
        if frame_first:
            results = id_first_to_frame_first(self.results)
            return results
        else:
            return self.results


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(
        self,
        pos,
        score,
        pred_ct,
        track_id,
        features,
        inactive_patience,
        max_features_num,
        mm_steps,
    ):
        self.id = track_id
        self.pos = pos
        self.pred_ct = pred_ct
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
        self.last_v = torch.Tensor([])
        self.gt_id = None

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        # print(self.max_features_num)
        # print(self.features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())


def frame_first_to_id_first(frame_first):
    """
    Frame first result: {Frame ID: a list of [x1, y1, x2, y2, score, ...]}
    Track first result: {Track ID: {Frame ID: [x1, y1, x2, y2, score, ...]}}
    """
    results = {}
    for frameid, bbs in frame_first.items():
        for one_bb in bbs:
            x1, y1, x2, y2, score, cur_id = (
                one_bb[0],
                one_bb[1],
                one_bb[2],
                one_bb[3],
                one_bb[4],
                one_bb[5],
            )
            if cur_id not in results:
                results[cur_id] = {}
            results[cur_id][frameid] = np.array([x1, y1, x2, y2, score])
    return results
