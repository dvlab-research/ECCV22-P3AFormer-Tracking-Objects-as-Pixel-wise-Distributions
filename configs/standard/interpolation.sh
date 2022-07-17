# MOT 15
EXP_DIR=exps/p3aformer_trained
EXP_ID=''
python3 interpolation.py \
     --dataset_name MOT15 \
     --data_dir /data/dataset/MOT15/ \
     --input_txt_dir ${EXP_DIR}/${EXP_ID}/txt \
     --output_txt_dir ${EXP_DIR}/${EXP_ID}/txt_interpolated

# MOT 17
EXP_DIR=exps/p3aformer_trained
EXP_ID='p3aformer_trained'
python3 interpolation.py \
     --dataset_name MOT17 \
     --data_dir /data/dataset/mot/ \
     --input_txt_dir ${EXP_DIR}/${EXP_ID}/txt \
     --output_txt_dir ${EXP_DIR}/${EXP_ID}/txt_interpolated