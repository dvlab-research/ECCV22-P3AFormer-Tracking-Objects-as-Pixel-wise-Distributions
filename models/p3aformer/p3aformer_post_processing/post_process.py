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
import cv2
from util.image import transform_preds_with_trans, get_affine_transform


def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)


def generic_post_process(opt, dets, c, s, h, w, filter_by_scores=0.3):
  if not ('scores' in dets):
    return [{}], [{}]
  ret = []

  for i in range(len(dets['scores'])):
    preds = []
    trans = get_affine_transform(
      c[i], s[i], 0, (w, h), inv=1).astype(np.float32)
    for j in range(len(dets['scores'][i])):
      if dets['scores'][i][j] < filter_by_scores:
        break

      item = {}
      item['score'] = dets['scores'][i][j]
      item['class'] = int(dets['clses'][i][j]) + 1
      item['ct'] = transform_preds_with_trans(
        (dets['cts'][i][j]).reshape(1, 2), trans).reshape(2)

      if 'tracking' in dets:
        # displacement to original image space
        tracking = transform_preds_with_trans(
          (dets['tracking'][i][j] + dets['cts'][i][j]).reshape(1, 2), trans).reshape(2)
        item['tracking'] = tracking - item['ct']  # ct in the ct int in original image plan
        item['pre_cts'] = tracking

      if 'bboxes' in dets:
        bbox = transform_preds_with_trans(
          dets['bboxes'][i][j].reshape(2, 2), trans).reshape(4)
        item['bbox'] = bbox

      preds.append(item)

    ret.append(preds)
  
  return ret