import os
from visualization_tool import Visualizer
import cv2
import pycocotools.coco as coco
import pdb
from collections import defaultdict

dataset_root = "/data/dataset/kitti"
ann_path = os.path.join(dataset_root, 'annotations', 'tracking_val_half.json')
img_dir = os.path.join(dataset_root, 'data_tracking_image_2/training', 'image_02')
output_path = "/data/cache"
os.makedirs(output_path, exist_ok=True)

coco_obj = coco.COCO(ann_path)
video_info = coco_obj.dataset['videos']
VidtoVname = {}
for v_info in video_info:
    VidtoVname[v_info['id']] = v_info['file_name']
video_to_images = defaultdict(list)
for image in coco_obj.dataset['images']:
    if image['video_id'] not in VidtoVname.keys():
        continue
    video_to_images[VidtoVname[image['video_id']]].append(image)
image_file_name_to_anns = defaultdict(list)
for anns in coco_obj.dataset['annotations']:
    image_file_name = coco_obj.dataset['images'][anns['image_id'] - 1]['file_name']
    image_file_name_to_anns[image_file_name].append(anns)
for video_id in video_to_images:
    print(f"Visualizing video: {video_id} ...")
    visualizer = Visualizer()
    for idx, image_d in enumerate(video_to_images[video_id]):
        img_path = os.path.join(img_dir, image_d['file_name'])
        assert os.path.exists(img_path), f"{img_path} does not exist!"
        img = cv2.imread(img_path)
        visualizer.add_img(img, img_id=idx)
        anns = image_file_name_to_anns[image_d['file_name']]
        for jdx, cur_anns in enumerate(anns):
            track_id = cur_anns['track_id']
            if int(cur_anns['category_id']) == 2:
                add_txt = "" #'_Car'
            elif int(cur_anns['category_id']) == 1:
                add_txt = "" #'_Pedestrian'
            else:
                continue
            bbox = [cur_anns['bbox'][0], cur_anns['bbox'][1], cur_anns['bbox'][0] + cur_anns['bbox'][2], 
             cur_anns['bbox'][1] + cur_anns['bbox'][3]]
            if track_id > 100000:
                track_id -= 100000
            visualizer.add_coco_bbox(bbox, 0, conf=track_id, add_txt=add_txt, img_id=idx)
            # pdb.set_trace()
        visualizer.save_video(path=output_path, name=video_id)
