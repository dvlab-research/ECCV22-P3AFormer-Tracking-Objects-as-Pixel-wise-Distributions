python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--dataset_name MOT17 --dataset_file crowdHuman --output_dir=./output/jul14_whole_ch_from_COCO --batch_size=1 \
--num_workers=4 --resume=./output/whole_coco/checkpoint0049.pth --pre_hm --tracking \
--data_dir=/data/dataset/crowdhuman --meta_arch p3aformer
