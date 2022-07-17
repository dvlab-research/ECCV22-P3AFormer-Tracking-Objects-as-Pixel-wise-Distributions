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

import torch
import torch.nn as nn
from util.image import gaussian_radius
import math
import numpy as np

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _sigmoid12(x):
  y = torch.clamp(x.sigmoid_(), 1e-12)
  return y

def _gather_feat(feat, ind):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  return feat

def _tranpose_and_gather_feat(feat, ind):
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.view(feat.size(0), -1, feat.size(3))
  feat = _gather_feat(feat, ind)
  return feat

def flip_tensor(x):
  return torch.flip(x, [3])
  # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def _nms(heat, kernel=3):
  pad = (kernel - 1) // 2

  hmax = nn.functional.max_pool2d(
      heat, (kernel, kernel), stride=1, padding=pad)
  keep = (hmax == heat).float()
  return heat * keep

def _topk_channel(scores, K=100):
  batch, cat, height, width = scores.size()
  
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys   = (topk_inds / width).int().float()
  topk_xs   = (topk_inds % width).int().float()

  return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=100):
  batch, cat, height, width = scores.size()

  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys   = (topk_inds / width).int().float()
  topk_xs   = (topk_inds % width).int().float()

  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  topk_inds = _gather_feat(
      topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_bis(scores, K=100):
  batch, cat, height, width = scores.size()

  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  origin_topK_inds = topk_inds

  topk_inds = topk_inds % (height * width)
  topk_ys = (topk_inds / width).int().float()
  topk_xs = (topk_inds % width).int().float()

  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  topk_inds = _gather_feat(
    topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs, topk_ind, origin_topK_inds


def _topk_soft(scores, hm_raw=None, K=100, softargmax=None):
  batch, cat, height, width = scores.size()
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  if softargmax is not None:
      soft_x = []
      soft_y = []
      for k in range(K):
          # argmax for b,cat
          # topk_inds is of shape (b,c, K)  K indexes max values for each category of each batch
          soft_out = softargmax(hm_raw, topk_inds[:, :, k].view(-1))
          # soft_out is of shape (b,c,2), soft argmax (x,y) for each category of each batch
          soft_x.append(soft_out[:, :, 0].unsqueeze(2))
          soft_y.append(soft_out[:, :, 1].unsqueeze(2))
      assert len(soft_x) == len(soft_y)
      if len(soft_x) !=0:
        # b,c,K
        soft_x = torch.cat(soft_x, dim=2)
        soft_y = torch.cat(soft_y, dim=2)

  topk_inds = topk_inds % (height * width)
  topk_ys = (topk_inds / width).int().float()
  topk_xs = (topk_inds % width).int().float()

  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  topk_inds = _gather_feat(
    topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  #soft pick by channels
  topk_ys_soft = _gather_feat(soft_y.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs_soft = _gather_feat(soft_x.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs, topk_ys_soft, topk_xs_soft


def pull_topk_soft(hm_raw=None, K=100, topk_inds=None, softargmax=None, wh=None, topk_ind=None):
  batch, cat, height, width = hm_raw.size()
  soft_x = []
  soft_y = []
  for k in range(K):
      # argmax for b,cat
      # topk_inds is of shape (b,c, K)  K indexes max values for each category of each batch
      radius = []  # one radius for one k of one batch
      for bth in range(wh.shape[0]):
        for categ in range(cat):
          box_w, box_h = wh[bth, k, 0].item(), wh[bth, k, 1].item()
          if (box_w > 0 and box_h > 0):
            radius.append(max(1, gaussian_radius((math.ceil(box_h), math.ceil(box_w)))))
            # print(radius)
          else:
            radius.append(2)
      soft_out = softargmax(hm_raw, topk_inds[:, :, k].view(-1), np.asarray(radius)*2+1)
      # soft_out is of shape (b,c,2), soft argmax (x,y) for each category of each batch
      soft_x.append(soft_out[:, :, 0].unsqueeze(2))
      soft_y.append(soft_out[:, :, 1].unsqueeze(2))
  assert len(soft_x) == len(soft_y)
  if len(soft_x) !=0:
    # b,c,K
    soft_x = torch.cat(soft_x, dim=2)
    soft_y = torch.cat(soft_y, dim=2)

  #soft pick by channels
  topk_ys_soft = _gather_feat(soft_y.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs_soft = _gather_feat(soft_x.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_ys_soft, topk_xs_soft