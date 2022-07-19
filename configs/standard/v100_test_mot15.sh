# validate our trained
# for MOT15
EXP_DIR=output/feb9_long_epoch
EXP_ID='mot17_p3aformer_long_epoch'
MODEL_NAME='checkpoint0199.pth'
python3 eval.py \
     --meta_arch p3aformer \
     --dataset_name MOT15 \
     --epoch 200 \
     --with_box_refine \
     --lr_drop 100 \
     --lr 2e-4 \
     --lr_backbone 2e-5 \
     --pretrained ${EXP_DIR}/${MODEL_NAME} \
     --output_dir ${EXP_DIR}/${EXP_ID} \
     --batch_size 1 \
     --sample_mode 'random_interval' \
     --sample_interval 10 \
     --sampler_steps 50 90 120 \
     --sampler_lengths 2 3 4 5 \
     --update_query_pos \
     --merger_dropout 0 \
     --dropout 0 \
     --random_drop 0.1 \
     --fp_ratio 0.3 \
     --query_interaction_layer 'QIM' \
     --extra_track_attn \
     --resume ${EXP_DIR}/${MODEL_NAME} \
     --mot_path datasets \
     --detr_path ${EXP_DIR}/${MODEL_NAME} \
     --reid_path ${EXP_DIR}/ResNet_iter_25245.pth \
     --data_dir=/data/dataset/MOT15/ \
     --track_thre 0.65 \
     --low_thre 0.2 \
     --first_assign_thre 0.9 \
     --second_assign_thre 0.5