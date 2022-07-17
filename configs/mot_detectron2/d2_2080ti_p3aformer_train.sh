export OUTPUT_DIR="output/jun3_2080ti"
python d2_main.py \
  --config-file configs/mot_detectron2/p3aformer_small.yaml \
  --num-gpus 8 SOLVER.IMS_PER_BATCH 16 SOLVER.MAX_ITER 83100 OUTPUT_DIR ${OUTPUT_DIR} INPUT.VAL_DATA_DIR "/data/dataset/mot" MODEL.DENSETRACK.ENC_LAYERS 2 MODEL.DENSETRACK.DEC_LAYERS 3