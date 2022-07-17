python d2_main.py \
  --config-file configs/mot_detectron2/p3aformer_small.yaml \
  --num-gpus 1 DATALOADER.NUM_WORKERS 0 DATASETS.TEST '("MOT17",)' INPUT.VAL_DATA_DIR "/data/dataset/mot" TRACK.VIS True