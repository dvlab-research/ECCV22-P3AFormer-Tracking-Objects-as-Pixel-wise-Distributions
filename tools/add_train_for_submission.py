import os
import shutil

mot_path = "/data/dataset/mot"
sub_dir = 'train'
seq_nums = os.listdir('/data/dataset/mot/train')
accs = []
seqs = []
predict_path = "/data/dataset/mot/train_result"
for seq_num in seq_nums:
    shutil.copyfile(os.path.join(mot_path, sub_dir, f'{seq_num}/gt/gt.txt'),
                    os.path.join(predict_path, f'{seq_num}.txt'))
