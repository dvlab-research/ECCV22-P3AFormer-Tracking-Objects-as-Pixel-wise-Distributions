from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
try:
  from .generic_dataset_train import GenericDataset
except:
  from generic_dataset_train import GenericDataset


class MOT17(GenericDataset):
  num_classes = 1
  num_joints = 17
  default_resolution = [640, 1088]
  max_objs = 300
  class_name = ['person']
  cat_ids = {1: 1}

  def __init__(self, opt, split):
    super(MOT17, self).__init__()
    data_dir = opt.data_dir
    if split == 'test':
      img_dir = os.path.join(
        data_dir, 'test')
    else:
      img_dir = os.path.join(
        data_dir, 'train')
    if opt.half_train:
      print("==> Using half of the MOT 17 data!")
    if split == 'train' and not opt.half_train:
      ann_path = os.path.join(data_dir, 'annotations_onlySDP', '{}.json').format(split)
    else:
      ann_path = os.path.join(data_dir, 'annotations_onlySDP', '{}_half.json').format(split)

    print('==> initializing MOT17 {} data.'.format(split))

    self.images = None
    # load image list and coco
    super(MOT17, self).__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def _save_results(self, records, fpath):
    with open(fpath,'w') as fid:
      for record in records:
        line = json.dumps(record)+'\n'
        fid.write(line)
    return fpath

  def convert_eval_format(self, all_bboxes):
    detections = []
    person_id = 1
    for image_id in all_bboxes:
      if type(all_bboxes[image_id]) != type({}):
        # newest format
        dtboxes = []
        for j in range(len(all_bboxes[image_id])):
          item = all_bboxes[image_id][j]
          if item['class'] != person_id:
            continue
          bbox = item['bbox']
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          bbox_out  = list(map(self._to_float, bbox[0:4]))
          detection = {
              "tag": 1,
              "box": bbox_out,
              "score": float("{:.2f}".format(item['score']))
          }
          dtboxes.append(detection)
      img_info = self.coco.loadImgs(ids=[image_id])[0]
      file_name = img_info['file_name']
      detections.append({'ID': file_name[:-4], 'dtboxes': dtboxes})
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    self._save_results(self.convert_eval_format(results),
                       '{}/results_crowdhuman.odgt'.format(save_dir))
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    try:
      os.system('python tools/crowdhuman_eval/demo.py ' + \
                '../data/crowdhuman/annotation_val.odgt ' + \
                '{}/results_crowdhuman.odgt'.format(save_dir))
    except:
      print('Crowdhuman evaluation not setup!')


def build(image_set, args):
  d = MOT17(args, image_set)
  # input output shapes
  args.input_h, args.input_w = d.default_resolution[0], d.default_resolution[1]
  args.output_h = args.input_h // args.down_ratio
  args.output_w = args.input_w // args.down_ratio
  args.input_res = max(args.input_h, args.input_w)
  args.output_res = max(args.output_h, args.output_w)
  # threshold
  args.out_thresh = max(args.track_thresh, args.out_thresh)
  args.pre_thresh = max(args.track_thresh, args.pre_thresh)
  args.new_thresh = max(args.track_thresh, args.new_thresh)
  args.adaptive_clip = True
  return d
