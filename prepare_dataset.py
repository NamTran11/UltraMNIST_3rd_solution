import os
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
np.random.seed(42)

# create merge dataset
list_txt_path = [
    # Path('./dataset/220320_30K_p_test/220320_30K_p_test.txt'),
    # Path('./dataset/220331_30K_train_tiny_v2/220331_30K_train_tiny_v2.txt'),
    # Path('./dataset/220331_30K_test_tiny_v2/220331_30K_test_tiny_v2.txt'), '220405_30K_test'
    # Path('./dataset/220405_60K_test_tiny_v2/220405_60K_test_tiny_v2.txt'),
    # Path('./dataset/220405_30K_test/220405_30K_test.txt'),
    # Path('./dataset/220405_30K_train/220405_30K_train.txt'),

    # Path('./dataset/pseudo/test/normal_ver1/normal_ver1.txt'),
    # Path('./dataset/pseudo/train/normal_0976/normal_0976.txt'),
    # Path('./dataset/pseudo/test/tiny_ver1/tiny_ver1.txt'),
    # Path('./dataset/pseudo/train/tiny_50/tiny_50.txt'),

    # Path('./dataset/220414_30K_train/220414_30K_train.txt'),
    # Path('./dataset/220414_30K_test/220414_30K_test.txt'),
    # Path('./dataset/pseudo/train/220413_0989/220413_0989.txt'),
    # Path('./dataset/pseudo/test/220413_09887/220413_09887.txt'),
    Path('./dataset/pseudo/train/220413_0989_ver2/220413_0989_ver2.txt'),
    Path('./dataset/pseudo/test/220413_0989_ver2/220413_0989_ver2.txt'),
]

list_file_path = []
for txt_path in list_txt_path:
    data_dir = txt_path.parent.joinpath(txt_path.stem)
    with open(str(txt_path), 'r') as f:
        file_names = [line.strip() for line in f.readlines()]
    for file_name in file_names:
        list_file_path.append(str(data_dir.joinpath(file_name + '.jpg')))
np.random.shuffle(list_file_path)

print('list_txt_path', list_txt_path, len(list_file_path))
save_dir_name = './convertor_220414_pseudo_v2'
if Path(save_dir_name).is_dir():
    shutil.rmtree(save_dir_name)
    print('REMOVE', save_dir_name)

# copy file
index = list(range(len(list_file_path)))
for fold in [0]:
    val_file_paths = list_file_path[len(index) * fold // 5: len(index) * (fold + 1) // 5]
    for file_path in tqdm(list_file_path, desc='Progress Bar'):
        if file_path in val_file_paths:
            path2save = 'val2017/'
        else:
            path2save = 'train2017/'
        # copy image and gt in to new folder
        image_dir = save_dir_name + '/fold{}/images/{}'.format(fold, path2save)
        gt_dir = save_dir_name + '/fold{}/labels/{}'.format(fold, path2save)
        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        image_path = Path(file_path)
        gt_path = image_path.with_suffix('.txt')
        shutil.copy(image_path, image_dir)
        shutil.copy(gt_path, gt_dir)
    print('process ok at fold_{}'.format(fold))
