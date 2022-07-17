from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from util.p3aformer.tracker_util import bbox_overlaps, warp_pos
import math
from util.p3aformer.p3aformer_misc import (
    get_flow,
    NestedTensor,
    gaussian_radius,
    affine_transform,
    draw_umich_gaussian,
    soft_nms_pytorch,
)
import lap
from tracker.common.track_structure_transfer import *


class Tracker:
    cl = 1

    def __init__(self, obj_detect, tracker_cfg, postprocessor=None, main_args=None):
        self.obj_detect = obj_detect
        self.detection_nms_thresh = tracker_cfg["detection_nms_thresh"]
        self.public_detections = tracker_cfg["public_detections"]
        self.inactive_patience = tracker_cfg["inactive_patience"]
        self.do_reid = tracker_cfg["do_reid"]
        self.max_features_num = tracker_cfg["max_features_num"]
        self.reid_sim_threshold = tracker_cfg["reid_sim_threshold"]
        self.reid_iou_threshold = tracker_cfg["reid_iou_threshold"]
        self.do_align = tracker_cfg["do_align"]
        self.motion_model_cfg = tracker_cfg["motion_model"]
        self.postprocessor = postprocessor
        self.main_args = main_args
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
        self.img_features = None
        self.encoder_pos_encoding = None
        self.transforms = transforms.ToTensor()
        self.last_image = None
        self.pre_sample = None
        self.sample = None
        self.pre_img_features = None
        self.pre_encoder_pos_encoding = None

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
                    self.motion_model_cfg["n_steps"]
                    if self.motion_model_cfg["n_steps"] > 0
                    else 1,
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
        outputs = self.obj_detect(
            samples=self.sample,
            pre_samples=self.pre_sample,
            features=self.img_features,
            pos=self.encoder_pos_encoding,
            pre_features=self.pre_img_features,
            pre_pos=self.pre_encoder_pos_encoding,
            pre_hm=pre_hm,
        )
        results = self.postprocessor(outputs, batch["orig_size"], filter_score=True)[0]
        out_scores, labels_out, out_boxes, pre_cts = (
            results["scores"],
            results["labels"],
            results["boxes"],
            results["pre_cts"],
        )
        filtered_idx = labels_out == 1
        return (
            out_boxes[filtered_idx, :].cuda(),
            out_scores[filtered_idx].cuda(),
            pre_cts[filtered_idx].cuda(),
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

    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""

        if self.im_index > 0:
            if self.do_reid:
                for t in self.inactive_tracks:
                    # todo check shape and format
                    t.pos = warp_pos(t.pos, self.flow)

    @torch.no_grad()
    def step(self, blob):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # get backbone features
        self.sample = NestedTensor(blob["image"].cuda(), blob["pad_mask"].cuda())
        self.img_features, self.encoder_pos_encoding = self.obj_detect.backbone(
            self.sample
        )
        if self.pre_img_features is None:
            self.pre_sample = NestedTensor(
                blob["pre_image"].cuda(), blob["pre_pad_mask"].cuda()
            )
            (
                self.pre_img_features,
                self.pre_encoder_pos_encoding,
            ) = self.obj_detect.backbone(self.pre_sample)
        blob["img"] = (
            blob["orig_img"].clone().float() / 255.0
        )  # TODO: consider deprecating this
        # detect
        raw_private_det_pos, raw_private_det_scores, _ = self.detect_tracking(blob)
        return raw_private_det_pos, raw_private_det_scores

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
