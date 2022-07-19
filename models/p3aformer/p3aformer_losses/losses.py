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
# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat, _nms, _topk
import torch.nn.functional as F
from util import box_ops

def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    ''' Reimplemented focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _only_neg_loss(pred, gt):
    gt = torch.pow(1 - gt, 4)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
    return neg_loss.sum()


class FastFocalLoss(nn.Module):
    '''
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    '''

    def __init__(self):
        super(FastFocalLoss, self).__init__()
        self.only_neg_loss = _only_neg_loss

    def forward(self, out, target, ind, mask, cat):
        '''
        Arguments:
          out, target: B x C x H x W
          ind, mask: B x M
          cat (category id for peaks): B x M
        '''
        neg_loss = self.only_neg_loss(out, target)
        pos_pred_pix = _tranpose_and_gather_feat(out, ind)  # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))  # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
                   mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return - neg_loss
        return - (pos_loss + neg_loss) / num_pos

#todo check
def loss_boxes(output, target):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
    """

    # b,2; 2= [h,w]
    target_sizes = target["output_size"]  # padded image size
    img_h, img_w = target_sizes.unbind(1)

    # [batch, max object]
    centers = target['ind']
    # [batch, max object]
    mask = target['boxes_mask']

    # todo to check
    # pred_shape = [b, # max boxes, (reg_w, reg_h, w, h)] == mask shape
    # target shape = [b, # max boxes, 4], 4 = cx, cy, w, h, normalized by output w, and output h.
    pred = _tranpose_and_gather_feat(output, target['ind'])

    # batch, max#objects, 4
    # print("pred shape", pred.shape)
    # print("mask shape", mask.shape)

    # (reg_w, reg_h, w, h)] => cx, cy, w,h
    c_x_t = centers % img_w[:, None]
    c_y_t = centers//img_w[:, None]
    pred[:, :, 0] += c_x_t
    pred[:, :, 1] += c_y_t

    # norm pred with output w, h
    pred[:, :, 0::2]/= img_w[:, None, None]
    pred[:, :, 1::2]/= img_h[:, None, None]

    # print("pred", pred)
    # print("target['boxes']", target['boxes'])
    # collect_non_mask_pred, tgt, #boxes, 4
    collect_pred = pred.view(-1, 4)[mask.view(-1) == 1, :]
    collect_tgt = target['boxes'].clone().view(-1, 4)[mask.view(-1) == 1, :]


    if collect_tgt.shape[0] > 0:
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(collect_pred),
            box_ops.box_cxcywh_to_xyxy(collect_tgt)))

        loss_giou = loss_giou.sum()/collect_tgt.shape[0]

        # unnorm predict box_w, box_h
        scale_fct = torch.stack([torch.ones_like(img_w), torch.ones_like(img_h), img_w, img_h], dim=1)
        pred *= scale_fct[:, None, :]
        target_boxes_unnorm = target['boxes'].clone() * scale_fct[:, None, :]

        # mean h of target and predicted

        # b, max #boxes, 4
        predict_boxes_h_mean = 0.5 * (pred[:, :, 3].detach() + target_boxes_unnorm[:, :, 3].detach())
        predict_boxes_h_mean = torch.clamp(predict_boxes_h_mean, min=1)
        # renorm target and predicted box_w, box_h by mean box_h of target and predicted
        target_boxes_unnorm[:, :, 2:] = target_boxes_unnorm[:, :, 2:] / predict_boxes_h_mean[:, :, None]
        pred[:, :, 2:] = pred[:, :, 2:] / predict_boxes_h_mean[:, :, None]
        loss_bboxes = F.l1_loss(pred * mask[:, :,  None], target_boxes_unnorm * mask[:, :,  None], reduction='sum')
    else:
        loss_bboxes = loss_giou = torch.sum(pred*0.0)

    return loss_bboxes / (mask.sum() + 1e-4), loss_giou


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, mask, ind, target):
        # output: B x F x H x W
        # ind: B x M
        # mask: B x M x F
        # target: B x M x F
        pred = _tranpose_and_gather_feat(output, ind)  # B x M x F
        loss = mask * self.bceloss(pred, target)
        loss = loss.sum() / (mask.sum() + 1e-4)
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res