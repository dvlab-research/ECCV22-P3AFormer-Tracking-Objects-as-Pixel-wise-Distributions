# debug, new dataset, reduced model size
python main.py \
   --meta_arch p3aformer \
   --data_dir /data/dataset/mot \
   --dataset_name MOT17 \
   --dataset_file p3aformer_mot \
   --batch_size=2  \
   --output_dir=./output/debug \
   --num_workers=20 \
   --resume="" \
   --pre_hm \
   --tracking \
   --same_aug_pre \
   --image_blur_aug \
   --lr 1e-4 \
   --lr_backbone_names ["backbone.0"] \
   --lr_backbone 2e-5 \
   --lr_linear_proj_names ['reference_points', 'sampling_offsets',] \
   --lr_linear_proj_mult 0.1 \
   --lr_drop 40 \
   --epochs 23 \
   --weight_decay 1e-4 \
   --clip_max_norm 0.1 \
   --backbone 'resnet50' \
   --position_embedding 'sine' \
   --num_feature_levels 3 \
   --enc_layers 2 \
   --dec_layers 2 \
   --dim_feedforward 1024 \
   --hidden_dim 256 \
   --shift 0.05 \
   --scale 0.05 \
   --rotate 0 \
   --flip 0.5 \
   --hm_disturb 0.05 \
   --lost_disturb 0.4 \
   --fp_disturb 0.1 \
   --track_thresh 0.3 \
   --new_thresh 0.3 \
   --ltrb_amodal_weight 0.1