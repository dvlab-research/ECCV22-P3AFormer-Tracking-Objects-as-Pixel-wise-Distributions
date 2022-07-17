all_path2labels = {}
print("Trying loading all files ...")
for label_path in dataset_train.label_files:
    if osp.isfile(label_path):
        labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6).tolist()
        all_path2labels[label_path] = labels0
    else:
        raise ValueError('invalid label path: {}'.format(label_path))
for label_path in dataset_val.label_files:
    if osp.isfile(label_path):
        labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6).tolist()
        all_path2labels[label_path] = labels0
    else:
        raise ValueError('invalid label path: {}'.format(label_path))
import json

json.dump(all_path2labels, open("datasets/data_path/mot.json", 'w'))