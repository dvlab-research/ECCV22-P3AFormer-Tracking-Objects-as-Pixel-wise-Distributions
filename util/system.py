import os
import shutil
import glob

def remove_files_under_folder(folder, select_str):
    files = glob.glob(os.path.join(folder, '*'))
    for f in files:
        if os.path.isdir(f):
            continue
        if select_str is not None and select_str in f:
            os.remove(f)
    return