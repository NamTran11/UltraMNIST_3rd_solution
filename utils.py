
import cv2
from pathlib import Path
import tqdm
import pandas as pd
import numpy as np
import shutil


def visualize_result_v1():
    image_path = './dataset/test/ffphzgpfun.jpeg'
    img = cv2.imread(image_path)
    txt_path = './txt_predicts/yolov5/220320_30K_tiny_dataset_v2_overlap_split_new_version/ffphzgpfun.txt'
    with open(str(txt_path), 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    h_img, w_img = img.shape[:2]
    for line in lines:
        cls, x_center, y_center, w, h, score = line.split()
        x_center, y_center, w, h = list(map(float, [x_center, y_center, w, h]))
        x_min = int(w_img * (x_center - 0.5 * w))
        y_min = int(h_img * (y_center - 0.5 * h))
        x_max = int(w_img * (x_center + 0.5 * w))
        y_max = int(h_img * (y_center + 0.5 * h))
        print(cls)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 10)
    cv2.imwrite('./temp/bbbbbb.jpg', img)


def visualize_result_v2():
    def txt_2_line_infos(txt_path):
        with open(str(txt_path), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        ret = []
        for line in lines:
            cls, x_center, y_center, w, h, score = line.split()
            x_center, y_center, w, h, score = list(map(float, [x_center, y_center, w, h, score]))
            cls = int(cls)
            ret.append([cls,
                        x_center - 0.5 * w,
                        y_center - 0.5 * h,
                        x_center + 0.5 * w,
                        y_center + 0.5 * h,
                        score])
        return ret
    image_dir = Path('./dataset/test')
    txt_dir1 = Path('./txt_predicts/yolov5/220320_30K_tiny_dataset_v2_overlap_split_16split')
    txt_dir2 = Path('./txt_predicts/yolov5/220320_with_30K_p_from_testset_new_version')
    image_save_dir = Path('./temp')
    resize_to_save = (1024, 1024)
    list_image_path = list(image_dir.glob('*'))[:10]
    for image_path in list_image_path:
        txt_path1 = txt_dir1.joinpath(image_path.name).with_suffix('.txt')
        txt_path2 = txt_dir2.joinpath(image_path.name).with_suffix('.txt')
        if txt_path1.is_file() and txt_path2.is_file():
            img = cv2.imread(str(image_path))
            line_info1 = txt_2_line_infos(txt_path1)
            line_info2 = txt_2_line_infos(txt_path2)
            # print('line_info2', line_info2)
            line_infos = line_info1 + line_info2

            h_img, w_img = img.shape[:2]
            for line in line_infos:
                cls, x_min, y_min, x_max, y_max, score = line
                x_min = int(w_img * x_min)
                y_min = int(h_img * y_min)
                x_max = int(w_img * x_max)
                y_max = int(h_img * y_max)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 10)
                cv2.putText(img,
                            str(cls) + ' - ' + str(score),
                            (x_min, y_min),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            save_img_path = image_save_dir.joinpath(image_path.name)
            # resize before save
            img = cv2.resize(img, resize_to_save)
            cv2.imwrite(str(save_img_path), img)
            print('save ok at', save_img_path)


def visualize_pseudo_data(
    image_dir,
    save_dir='./temp',
    txt_dir=None,
    resize_to_save=(1024, 1024)
):
    def txt_2_line_infos(txt_path):
        with open(str(txt_path), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        ret = []
        for line in lines:
            cls, x_center, y_center, w, h = line.split()
            x_center, y_center, w, h = list(map(float, [x_center, y_center, w, h]))
            cls = int(cls)
            ret.append([cls,
                        x_center - 0.5 * w,
                        y_center - 0.5 * h,
                        x_center + 0.5 * w,
                        y_center + 0.5 * h])
        return ret
    image_save_dir = Path(save_dir)
    image_dir = Path(image_dir)
    txt_dir = Path(txt_dir) if txt_dir else image_dir
    list_image_path = list(image_dir.glob('*.jpg'))[:10]
    for image_path in list_image_path:
        txt_path = image_path.with_suffix('.txt')
        if txt_path.is_file():
            img = cv2.imread(str(image_path))
            line_infos = txt_2_line_infos(txt_path)
            h_img, w_img = img.shape[:2]
            for line in line_infos:
                cls, x_min, y_min, x_max, y_max = line
                x_min = int(w_img * x_min)
                y_min = int(h_img * y_min)
                x_max = int(w_img * x_max)
                y_max = int(h_img * y_max)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
                cv2.putText(img,
                            str(cls),
                            (x_min, y_min),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            save_img_path = image_save_dir.joinpath(image_path.name)
            # resize before save
            img = cv2.resize(img, resize_to_save)
            cv2.imwrite(str(save_img_path), img)
            print('save ok at', save_img_path)


def visualize_invalid_names(invalid_infos,
                            img_dir,
                            txt_dir,
                            image_save_dir='./temp',
                            resize_to_save=(1024, 1024),
                            num_visualize=10):
    def txt_2_line_infos(txt_path):
        with open(str(txt_path), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        ret = []
        for line in lines:
            cls, x_center, y_center, w, h, score = line.split()
            x_center, y_center, w, h, score = list(map(float, [x_center, y_center, w, h, score]))
            cls = int(cls)
            ret.append([cls,
                        x_center - 0.5 * w,
                        y_center - 0.5 * h,
                        x_center + 0.5 * w,
                        y_center + 0.5 * h,
                        score])
        return ret
    img_dir = Path(img_dir)
    txt_dir = Path(txt_dir)
    image_save_dir = Path(image_save_dir)

    np.random.shuffle(invalid_infos)
    invalid_infos = invalid_infos[:num_visualize]
    for info in invalid_infos:
        name, gt_sum, predict = info
        txt_path = txt_dir.joinpath(name + '.txt')
        image_path = img_dir.joinpath(name + '.jpeg')

        if txt_path.is_file():
            img = cv2.imread(str(image_path))
            line_infos = txt_2_line_infos(txt_path)

            h_img, w_img = img.shape[:2]
            for line in line_infos:
                cls, x_min, y_min, x_max, y_max, score = line
                x_min = int(w_img * x_min)
                y_min = int(h_img * y_min)
                x_max = int(w_img * x_max)
                y_max = int(h_img * y_max)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 10)
                cv2.putText(img,
                            str(cls) + '-' + str(score),
                            (x_min, y_min),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(img,
                        'gt_sum - ' + str(gt_sum) + '-' + str(predict),
                        (5, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3, cv2.LINE_AA)
            save_img_path = image_save_dir.joinpath(image_path.name)
            # resize before save
            img_org = cv2.imread(str(image_path))
            img = cv2.resize(img, resize_to_save)
            img_org = cv2.resize(img_org, resize_to_save)
            cv2.imwrite(str(save_img_path), np.hstack((img, img_org)))
            print('save ok at', save_img_path)


def create_pseudo_data_from_train(
    valid_infos,
    img_dir='./dataset/test',
    txt_dir='./temp',
    data_save_dir='./temp',
    resize_to_save=(1024, 1024),
    tiny_mode=False,
    max_tiny_size=30,
):

    def get_yolo_rect(points, output_size):
        h_output, w_output = output_size
        x_min = min(p[0] for p in points)
        y_min = min(p[1] for p in points)
        x_max = max(p[0] for p in points)
        y_max = max(p[1] for p in points)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        return [x_center / w_output, y_center / h_output, w / w_output, h / h_output]

    def is_inner_cut_box(cut_box, points, overlap_ratio=0.7, size_thresh=None):
        x1_c, y1_c, x2_c, y2_c = cut_box
        x_min = min(p[0] for p in points)
        y_min = min(p[1] for p in points)
        x_max = max(p[0] for p in points)
        y_max = max(p[1] for p in points)
        if isinstance(size_thresh, int):
            # ignore all medium and large digit
            if max(x_max - x_min, y_max - y_min) >= size_thresh:
                return False, None
        xA = max(x1_c, x_min)
        yA = max(y1_c, y_min)
        xB = min(x2_c, x_max)
        yB = min(y2_c, y_max)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        point_area = (x_max - x_min + 1) * (y_max - y_min + 1)
        # return interArea / point_area > overlap_ratio
        if interArea / point_area > overlap_ratio:
            return True, [[x_min - x1_c, y_min - y1_c], [x_max - x1_c, y_max - y1_c]]
        return False, None

    img_dir = Path(img_dir)
    txt_dir = Path(txt_dir)
    data_save_dir = Path(data_save_dir)
    if not data_save_dir.is_dir():
        data_save_dir.mkdir(parents=True)

    data_save_dir_img = data_save_dir.joinpath(data_save_dir.name)
    if not data_save_dir_img.is_dir():
        data_save_dir_img.mkdir(parents=True)

    list_image_name = []
    for info_idx, info in enumerate(valid_infos):
        name, gt_sum, predict = info
        txt_path = txt_dir.joinpath(name + '.txt')
        image_path = img_dir.joinpath(name + '.jpeg')

        if txt_path.is_file():
            txt_save_path = data_save_dir_img.joinpath(txt_path.name)
            image_save_path = txt_save_path.with_suffix('.jpg')

            img = cv2.imread(str(image_path))

            with open(str(txt_path), 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            ret = []
            for line in lines:
                cls, x_center, y_center, w, h, score = line.split()
                ret.append(' '.join([cls, x_center, y_center, w, h]))
            if not tiny_mode:
                img = cv2.resize(img, resize_to_save)
                cv2.imwrite(str(image_save_path), img)
                with open(str(txt_save_path), 'w') as f:
                    f.write('\n'.join(ret))
                list_image_name.append(str(image_save_path.stem))
                print('save normal ok at: {}/{}'.format(info_idx, len(valid_infos)))
            else:
                h_img, w_img = img.shape[:2]
                cdd_boxs = []  # list of all tiny digit
                for line in lines:
                    cls, x_center, y_center, w, h, score = line.split()
                    x_center, y_center, w, h, score = list(map(float, [x_center, y_center, w, h, score]))
                    cls = int(cls)
                    x_min = int(w_img * (x_center - 0.5 * w))
                    y_min = int(h_img * (y_center - 0.5 * h))
                    x_max = int(w_img * (x_center + 0.5 * w))
                    y_max = int(h_img * (y_center + 0.5 * h))
                    if max(x_max - x_min, y_max - y_min) <= max_tiny_size:
                        cdd_boxs.append([cls, x_min, y_min, x_max, y_max])
                for idx_box, cdd_box in enumerate(cdd_boxs):
                    croped_txt_save_path = data_save_dir_img.joinpath(txt_path.stem + '_' + str(idx_box) + '.txt')
                    croped_image_save_path = data_save_dir_img.joinpath(txt_path.stem + '_' + str(idx_box) + '.jpg')

                    glob_h, glob_w = img.shape[:2]
                    out_h, out_w = resize_to_save

                    # pick location to crop
                    cls, x_min_c, y_min_c, x_max_c, y_max_c = cdd_box
                    post_x = np.random.randint(max(0, x_max_c - out_w), min(x_min_c, glob_w - out_w))
                    post_y = np.random.randint(max(0, y_max_c - out_h), min(y_min_c, glob_h - out_h))

                    cut_box = [post_x, post_y, post_x + out_w, post_y + out_h]
                    image_croped = img[post_y: post_y + out_h, post_x: post_x + out_w]
                    cv2.imwrite(str(croped_image_save_path), image_croped)

                    # check is inner and save as yoloformat
                    new_lines = []
                    for cdd_box_j in cdd_boxs:
                        cls1, x_min_1, y_min_1, x_max_1, y_max_1 = cdd_box_j
                        is_inner, new_points = is_inner_cut_box(
                            cut_box,
                            [[x_min_1, y_min_1], [x_max_1, y_max_1]],
                            overlap_ratio=0.7,
                            size_thresh=max_tiny_size)
                        if is_inner:
                            box_info = list(map(str, get_yolo_rect(new_points, resize_to_save)))
                            new_lines.append(' '.join([str(cls1)] + box_info))
                    list_image_name.append(str(croped_image_save_path.stem))
                    with open(str(croped_txt_save_path), 'w') as f:
                        f.write('\n'.join(new_lines))
                print('save tiny ok at: {}/{}'.format(info_idx, len(valid_infos)))
    save_list_name_path = str(data_save_dir_img) + '.txt'
    print(list_image_name)
    with open(str(save_list_name_path), 'w') as f:
        f.write('\n'.join(list_image_name))
    print('FINISHED GENERATION')


def convert_coord():
    input_dir = Path('./txt_predicts/yolov5_in_trainset/220320_with_30K_p_from_testset_predict_in_trainset')
    output_dir = Path('./txt_predicts/yolov5_in_trainset/220320_with_30K_p_from_testset_predict_in_trainset_new_version')
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    for txt_path in input_dir.glob('*.txt'):
        save_path = output_dir.joinpath(txt_path.name)
        with open(str(txt_path), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        new_lines = []
        for line in lines:
            cls, x_center, y_center, w, h, score = line.split()
            x_center, y_center, w, h = list(map(float, [x_center, y_center, w, h]))
            x_center = x_center * 1024 / 4000
            y_center = y_center * 1024 / 4000
            w = w * 1024 / 4000
            h = h * 1024 / 4000
            x_center, y_center, w, h = list(map(str, [x_center, y_center, w, h]))
            new_lines.append(' '.join([cls, x_center, y_center, w, h, score]))
        with open(str(save_path), 'w') as f:
            f.write('\n'.join(new_lines))


def compare_csv():
    def csv2dict(csv_path):
        ret = {}
        with open(str(csv_path), 'r') as f:
            lines = [line.strip() for line in f.readlines()[1:]]
        for line in lines:
            name, s = line.split(',')
            ret[name] = int(s)
        return ret

    # for test set
    csv_path1 = './final_csv_files/220414_8merge_ver5_four_weight_normal/submission_wfb_1_1_1_1_1_1_1.csv'
    csv_path2 = './final_csv_files/220414_8merge_ver5_four_weight_normal_v2/submission_wfb_1_1_1_1_1_1_1.csv'

    # # for train set
    # csv_path1 = './dataset/train.csv'
    # csv_path2 = './final_csv_files/yolov5_in_trainset/220327_60K_tiny_dataset_v2_05_9_16_normal/submission_wfb_1_1_1_1_1_1_1.csv'

    invalid_names = []
    valid_names = []
    dict1 = csv2dict(csv_path1)
    dict2 = csv2dict(csv_path2)
    correct = 0
    for k, v in dict1.items():
        if dict2[k] == v:
            correct += 1
            valid_names.append([str(k), v, dict2[k]])
        else:
            invalid_names.append([str(k), v, dict2[k]])
    print(correct)
    print(correct / len(dict1))
    return invalid_names, valid_names


if __name__ == '__main__':
    invalid_names, valid_names = compare_csv()
    print(len(invalid_names + valid_names))
    # visualize_invalid_names(
    #     invalid_infos=invalid_names,
    #     img_dir='./dataset/train',
    #     txt_dir='./txt_predicts/yolov5_in_trainset/220327_60K_tiny_dataset_v2_05_9_16_normal/1_1_1_1_1_1_1',
    #     image_save_dir='./temp',
    #     resize_to_save=(2048, 2048),
    #     num_visualize=30)

    # convert_coord()
    # visualize_result_v2()

    # create_pseudo_data_from_train(
    #     valid_infos=valid_names + invalid_names,
    #     img_dir='./dataset/test',
    #     # txt_dir='./txt_predicts/yolov5/220331_60K_normal_tiny_v2_05_9split_07_16split_07_25split_06_normal_085_640_e50_ext01_pp/1_1_1_1_1',
    #     # txt_dir='./txt_predicts/yolov5/220412_8merge_ver5_three_weight/1_1_1_1_1_1_1',
    #     txt_dir='./txt_predicts/yolov5/220414_7merge_ver5_three_weight/1_1_1_1_1_1_1',
    #     data_save_dir='./dataset/pseudo/test/220413_0989_ver2',
    #     resize_to_save=(1024, 1024),
    #     tiny_mode=False,
    #     # max_tiny_size=50,
    # )
    # visualize_pseudo_data(image_dir='./dataset/pseudo/test/temp_10/temp_10', save_dir='./temp')
