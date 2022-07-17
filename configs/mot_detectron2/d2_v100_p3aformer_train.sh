export OUTPUT_DIR="output/June2Mixed"
python d2_main.py \
  --config-file configs/mot_detectron2/p3aformer_big.yaml \
  --num-gpus 4 SOLVER.IMS_PER_BATCH 16 SOLVER.MAX_ITER 83100 OUTPUT_DIR ${OUTPUT_DIR} MODEL.DENSETRACK.ENC_LAYERS 5 MODEL.DENSETRACK.DEC_LAYERS 5