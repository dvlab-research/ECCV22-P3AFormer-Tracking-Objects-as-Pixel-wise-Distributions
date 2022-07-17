import os
from visualization_tool import Visualizer
import cv2
import pycocotools.coco as coco
import pdb
from collections import defaultdict

dataset_root = "/data/dataset/mot"
ann_path = os.path.join(dataset_root, "annotations", "val_half.json")
img_dir = os.path.join(dataset_root, "train")
output_path = "/data/cache"
os.makedirs(output_path, exist_ok=True)

coco_obj = coco.COCO(ann_path)
video_info = coco_obj.dataset["videos"]
VidtoVname = {}
for v_info in video_info:
    VidtoVname[v_info["id"]] = v_info["file_name"]
video_to_images = defaultdict(list)
for image in coco_obj.dataset["images"]:
    if image["video_id"] not in VidtoVname.keys():
        continue
    video_to_images[VidtoVname[image["video_id"]]].append(image)
image_id_to_filename = {}
for one_image in coco_obj.dataset["images"]:
    image_id_to_filename[one_image["id"]] = one_image["file_name"]

image_file_name_to_anns = defaultdict(list)
for anns in coco_obj.dataset["annotations"]:
    image_file_name = image_id_to_filename[anns["image_id"]]
    image_file_name_to_anns[image_file_name].append(anns)
for video_id in video_to_images:
    print(f"Visualizing video: {video_id} ...")
    visualizer = Visualizer()
    for idx, image_d in enumerate(video_to_images[video_id]):
        print(f"Stepping frame {idx} / {len(video_to_images[video_id])} ...")
        img_path = os.path.join(img_dir, image_d["file_name"])
        assert os.path.exists(img_path), f"{img_path} does not exist!"
        img = cv2.imread(img_path)
        visualizer.add_img(img, img_id=idx)
        anns = image_file_name_to_anns[image_d["file_name"]]
        for jdx, cur_anns in enumerate(anns):
            track_id = cur_anns["track_id"]
            bbox = [
                cur_anns["bbox"][0],
                cur_anns["bbox"][1],
                cur_anns["bbox"][0] + cur_anns["bbox"][2],
                cur_anns["bbox"][1] + cur_anns["bbox"][3],
            ]
            if track_id > 100000:
                track_id -= 100000
            visualizer.add_coco_bbox(bbox, 0, conf=track_id, add_txt="", img_id=idx)
        visualizer.save_video(path=output_path, name=video_id)
        
