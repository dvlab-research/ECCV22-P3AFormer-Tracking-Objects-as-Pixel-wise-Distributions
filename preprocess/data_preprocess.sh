python3 preprocess/convert_mot17_to_coco.py
python3 preprocess/convert_mot20_to_coco.py
python3 preprocess/convert_crowdhuman_to_coco.py
python3 preprocess/convert_cityperson_to_coco.py
python3 preprocess/convert_ethz_to_coco.py

bash preprocess/make_mixed_dirs.sh
python3 preprocess/mix_data_ablation.py
python3 preprocess/mix_data_test_mot17.py
python3 preprocess/mix_data_test_mot20.py