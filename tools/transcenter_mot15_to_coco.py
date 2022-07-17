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
import os
import numpy as np
import json

DATA_PATH = '/data/dataset/MOT15/'
OUT_PATH = DATA_PATH + 'annotations/'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
SPLITS = ['train_half', 'val_half', 'train', 'test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True

if __name__ == '__main__':
  for split in SPLITS:
    # TODO: the following would use train for test when HALF_VIDEO=True!! Check this!
    data_path = DATA_PATH + 'images/' + (split if not HALF_VIDEO else 'train')
    out_path = OUT_PATH + '{}.json'.format(split)
    out = {'images': [], 'annotations': [], 
           'categories': [{'id': 1, 'name': 'pedestrian'}],
           'videos': []}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for seq in sorted(seqs):
      if '.DS_Store' in seq:
        continue
      if 'MOT17' in DATA_PATH and 'SDP' not in seq:
        continue
      video_cnt += 1
      out['videos'].append({
        'id': video_cnt,
        'file_name': seq})
      seq_path = '{}/{}/'.format(data_path, seq)
      img_path = seq_path + 'img1/'
      ann_path = seq_path + 'gt/gt.txt'
      images = os.listdir(img_path)
      num_images = len([image for image in images if 'jpg' in image])
      if HALF_VIDEO and ('half' in split):
        image_range = [0, num_images // 2] if 'train' in split else \
          [int(0.75*num_images+0.5), num_images - 1]
      else:
        image_range = [0, num_images - 1]
      for i in range(num_images):
        if (i < image_range[0] or i > image_range[1]):
          continue
        image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1 - image_range[0],
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        out['images'].append(image_info)
      print('{}: {} images'.format(seq, num_images))
      if split != 'test':
        det_path = seq_path + 'det/det.txt'
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
        if CREATE_SPLITTED_ANN and ('half' in split):
          anns_out = np.array([anns[i] for i in range(anns.shape[0]) if \
            int(anns[i][0]) - 1 >= image_range[0] and \
            int(anns[i][0]) - 1 <= image_range[1]], np.float32)
          anns_out[:, 0] -= image_range[0]
          gt_out = seq_path + '/gt/gt_{}.txt'.format(split)
          fout = open(gt_out, 'w')
          for o in anns_out:
            fout.write(
              '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
              int(o[0]),int(o[1]),int(o[2]),int(o[3]),int(o[4]),int(o[5]),
              int(o[6]),int(o[7]),o[8]))
          fout.close()
        if CREATE_SPLITTED_DET and ('half' in split):
          dets_out = np.array([dets[i] for i in range(dets.shape[0]) if \
            int(dets[i][0]) - 1 >= image_range[0] and \
            int(dets[i][0]) - 1 <= image_range[1]], np.float32)
          dets_out[:, 0] -= image_range[0]
          det_out = seq_path + '/det/det_{}.txt'.format(split)
          dout = open(det_out, 'w')
          for o in dets_out:
            dout.write(
              '{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
              int(o[0]),int(o[1]),float(o[2]),float(o[3]),float(o[4]),float(o[5]),
              float(o[6])))
          dout.close()

        print(' {} ann images'.format(int(anns[:, 0].max())))
        for i in range(anns.shape[0]):
          frame_id = int(anns[i][0])
          if (frame_id - 1 < image_range[0] or frame_id - 1> image_range[1]):
            continue
          track_id = int(anns[i][1])
          cat_id = int(anns[i][7])
          ann_cnt += 1
          iscrowd = 0
          if not ('15' in DATA_PATH):
            if not (float(anns[i][8]) >= 0.1):
              iscrowd = 1
            if not (int(anns[i][6]) == 1):
              continue
            if (int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]): # Non-person
              continue
            if (int(anns[i][7]) in [2, 7, 8, 12]): # Ignored person
              category_id = -1
            else:
              category_id = 1
          else:
            category_id = 1
          ann = {'id': ann_cnt,
                 'iscrowd': int(iscrowd),
                 'category_id': category_id,
                 'image_id': image_cnt + frame_id,
                 'track_id': track_id,
                 'bbox': anns[i][2:6].tolist(),
                 'area': float(anns[i][4]*anns[i][5]),
                 'conf': float(anns[i][6])}
          out['annotations'].append(ann)
      image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(
      split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
        
        

