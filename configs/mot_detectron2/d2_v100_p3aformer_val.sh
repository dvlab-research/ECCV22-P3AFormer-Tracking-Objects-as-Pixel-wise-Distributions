# # validation MOT15
MODEL_NAME="output/feb15_v100/model_final.pth"
python d2_main.py \
  --config-file configs/mot_detectron2/p3aformer_small.yaml --eval-only  \
  --num-gpus 1 DATALOADER.NUM_WORKERS 0 SOLVER.IMS_PER_BATCH 1 SOLVER.MAX_ITER 83100 MODEL.WEIGHTS ${MODEL_NAME} INPUT.VAL_DATA_DIR "/data/dataset/MOT15" MODEL.DENSETRACK.ENC_LAYERS 6 MODEL.DENSETRACK.DEC_LAYERS 6

# open visualization on MOT17
MODEL_DIR="output/June2Mixed"
SPLIT="val_half"
MODEL_NAME=${MODEL_DIR}"/model_final.pth"
OUTPUT_DIR="output/June2Mixed/model_final"
python d2_main.py \
  --config-file configs/mot_detectron2/p3aformer_big.yaml --eval-only  \
  --num-gpus 1 DATALOADER.NUM_WORKERS 0 SOLVER.IMS_PER_BATCH 1 MODEL.WEIGHTS ${MODEL_NAME} INPUT.VAL_DATA_DIR "/data/dataset/mot" DATASETS.TEST '("MOT17",)' MODEL.DENSETRACK.ENC_LAYERS 5 MODEL.DENSETRACK.DEC_LAYERS 5 OUTPUT_DIR ${OUTPUT_DIR}
    # TRACK.VIS True 
