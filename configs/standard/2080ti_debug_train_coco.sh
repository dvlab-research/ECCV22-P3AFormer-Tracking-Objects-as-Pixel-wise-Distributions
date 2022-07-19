python main.py \
--dataset_name MOT17 --dataset_file coco \
--output_dir=./output/jul19_whole_coco --batch_size=1 --num_workers=0 --pre_hm --tracking --data_dir=/data/dataset/coco --scale 0.05 --shift 0.05 --flip 0.5 --meta_arch p3aformer --resume="" --num_feature_levels 3 --enc_layers 2 --dec_layers 2