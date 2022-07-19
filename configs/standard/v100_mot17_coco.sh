python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--dataset_name MOT17 --dataset_file coco \
--output_dir=./output/jul19_whole_coco --batch_size=3 --num_workers=20 --pre_hm \
--tracking --data_dir=/data/dataset/coco --scale 0.05 --shift 0.05 --flip 0.5 --meta_arch p3aformer --resume=""