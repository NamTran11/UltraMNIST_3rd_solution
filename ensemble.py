from audioop import reverse
import csv
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import itertools


def filter_inner_box_infos(box_infos, area_thresh=1.1):
    def is_invalid_box(box1, box2):
        cls1, x_min1, y_min1, x_max1, y_max1, score1 = box1
        cls2, x_min2, y_min2, x_max2, y_max2, score2 = box2
        # check box1 is inner box2
        if x_min1 >= x_min2 and x_max1 <= x_max2 and \
                y_min1 >= y_min2 and y_max1 <= y_max2:
            area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
            area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
            if area2 / area1 >= area_thresh:
                return True
        return False

    # sort by area
    box_infos.sort(key=lambda x: (x[3] - x[1]) * (x[4] - x[2]))
    invalid_idxs = []
    for i in range(len(box_infos)):
        for j in range(i + 1, len(box_infos)):
            if is_invalid_box(box_infos[i], box_infos[j]):
                invalid_idxs.append(i)
                break
    for idx in invalid_idxs[::-1]:
        del box_infos[idx]
    return box_infos


def filter_size_box_infos(box_infos, min_size=4, image_size=4000):
    # sort by area
    box_infos.sort(key=lambda x: (x[3] - x[1]) * (x[4] - x[2]))
    invalid_idxs = []
    for i in range(len(box_infos)):
        cls, x_min, y_min, x_max, y_max, score, _ = box_infos[i]
        w = int((x_max - x_min) * image_size)
        h = int((y_max - y_min) * image_size)
        if max(w, h) < min_size:
            invalid_idxs.append(i)
    invalid_idxs = list(set(invalid_idxs))
    invalid_idxs.sort()
    for idx in invalid_idxs[::-1]:
        del box_infos[idx]
    return box_infos


def preprocess_box_infos(box_infos, expand_ratio=0.1):
    # expand_box_info:
    for idx in range(len(box_infos)):
        cls, x_min, y_min, x_max, y_max, score, model_idx = box_infos[idx]
        w_ = x_max - x_min
        h_ = y_max - y_min
        new_x_min = x_min - expand_ratio * w_
        new_x_max = x_max + expand_ratio * w_
        new_y_min = y_min - expand_ratio * h_
        new_y_max = y_max + expand_ratio * h_
        box_infos[idx] = [cls, new_x_min, new_y_min, new_x_max, new_y_max, score, model_idx]
    return box_infos


def filter_overlap(box_infos):
    def get_overlap_info(box1, box2):
        cls1, x_min1, y_min1, x_max1, y_max1, score1, _ = box1
        cls2, x_min2, y_min2, x_max2, y_max2, score2, _ = box2
        if max(x_min1, x_min2) < min(x_max1, x_max2) and \
                max(y_min1, y_min2) < min(y_max1, y_max2):
            return True, cls1 if score1 > score2 else cls2
        return False, None

    # box_infos = filter_box_infos(box_infos)
    box_infos = filter_size_box_infos(box_infos, min_size=6)  # 7
    box_infos = preprocess_box_infos(box_infos, expand_ratio=0.075)  # 0.1
    # box_infos = filter_size_box_infos(box_infos, min_size=7)  # 7
    # box_infos = preprocess_box_infos(box_infos, expand_ratio=0.1)  # 0.1
    box_infos.sort(key=lambda x: x[5], reverse=True)  # sort by confidence
    valid_box_infos = []
    travel_idxs = []
    list_valid_number = []
    for i in range(len(box_infos)):
        if i in travel_idxs:
            continue  # skip
        travel_idxs.append(i)
        for j in range(i + 1, len(box_infos)):
            if j in travel_idxs:
                continue
            is_overlap__, cls_ = get_overlap_info(box_infos[i], box_infos[j])
            if is_overlap__:
                travel_idxs.append(j)
        list_valid_number.append(int(box_infos[i][0]))
        valid_box_infos.append(box_infos[i])
    list_valid_number = list(map(int, list_valid_number))
    # S1
    # ret = sum(list_valid_number)

    # S2
    # # sort by confidence
    # valid_box_infos.sort(key=lambda x: float(x[5]), reverse=True)
    # ret = sum([int(info[0]) for info in valid_box_infos[:5]])

    # S3
    invalid_idxs = []
    for idx, box_info in enumerate(valid_box_infos):
        cls, x_min, y_min, x_max, y_max, score, model_idx = box_info
        w_ = x_max - x_min
        h_ = y_max - y_min
        # if cls == 1 and h_ / w_ < 1.1:
        if cls == 1 and h_ / w_ < 2.0 and score < 0.8:  # 0.9712
            invalid_idxs.append(idx)
        # if max(w_, h_) < 8 and score < 0.8:
        #     invalid_idxs.append(idx)

    invalid_idxs = list(set(invalid_idxs))
    invalid_idxs.sort()
    for idx in invalid_idxs[::-1]:
        del valid_box_infos[idx]

    # if len(valid_box_infos) == 6:
    #     valid_box_infos.sort(key=lambda x: x[5], reverse=True)
    #     if valid_box_infos[5][5] == 1:
    #         ret = sum([int(info[0]) for info in valid_box_infos[:5]])
    #     else:
    #         ret = sum([int(info[0]) for info in valid_box_infos])
    ret = sum([int(info[0]) for info in valid_box_infos])

    if ret > 27:
        ret = 10
    return ret, valid_box_infos


def convert2csv(input_dir, save_csv_path='./submission.csv', method=1):
    save_csv_path = Path(save_csv_path)
    parent_path = save_csv_path.parent
    if not parent_path.is_dir():
        parent_path.mkdir(parents=True)
    print('__[method]__:', method)
    header = ['id', 'digit_sum']
    data_lines = []
    input_dir = Path(input_dir)
    for txt_path in tqdm(input_dir.glob('*.txt'), desc='Progress Bar'):
        # for txt_path in input_dir.glob('*.txt'):
        with open(str(txt_path), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        if method == 1:
            # version1: get digit_sum --- 0.54428
            # ver1: Single predict yolov5 220317 with 10K first generated data --- 0.54428
            digit_sum = sum(int(line.strip()[0]) for line in lines)
        elif method == 2:
            # version 2: --- 0.42485
            # ver2: Single predict yolov5 220317 with 10K first generated data --- 0.42485
            digit_infos = []
            for line in lines:
                es = line.split()
                digit_infos.append([int(es[0]), float(es[5])])
            # check confidence
            digit_infos = [info for info in digit_infos if float(info[1]) > 0.8]
            # sort by confidence
            digit_infos.sort(key=lambda x: float(x[1]), reverse=True)
            digit_sum = sum([int(info[0]) for info in digit_infos[:5]])
        elif method == 3:
            # version 3: check overlap and remove it
            box_infos = []
            for line in lines:
                class_idx, x_center, y_center, w, h, score = line.split()
                x_center, y_center, w, h, score = list(map(float, [x_center, y_center, w, h, score]))
                class_idx = int(class_idx)
                box_infos.append([
                    class_idx, x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w, y_center + 0.5 * h, score
                ])
            digit_sum, _ = filter_overlap(box_infos)
            # print('digit_sum', digit_sum)

        if digit_sum > 27:
            digit_sum = 26
        data_lines.append([txt_path.stem, digit_sum])
    with open(str(save_csv_path), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write multiple rows
        writer.writerows(data_lines)


def get_acc_by_file(
    # data_lines_dict,
    # valid_box_infos,
    file_name,
    list_input_dir,
    list_weights,
    list_conf_thresh,
    wfb=True,
):
    data_lines_dict = {'_'.join([str(e) for e in weights]): [] for weights in list_weights}
    valid_box_infos = {'_'.join([str(e) for e in weights]): {} for weights in list_weights}
    boxes_list = []
    scores_list = []
    labels_list = []
    dir_list = []
    iou_thr = 0.5
    skip_box_thr = 0.0001
    for idx_dir, input_dir in enumerate(list_input_dir):
        txt_path = input_dir.joinpath(file_name)
        if not txt_path.is_file():
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
            continue
        boxes = []
        scores = []
        labels = []
        dirs = []
        with open(str(txt_path), 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        for line in lines:
            class_idx, x_center, y_center, w, h, score = line.split()
            x_center, y_center, w, h, score = list(map(float, [x_center, y_center, w, h, score]))
            class_idx = int(class_idx)
            if score < list_conf_thresh[idx_dir]:
                continue
            boxes.append([x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w, y_center + 0.5 * h])
            scores.append(score)
            labels.append(class_idx)
            dirs.append(idx_dir)

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)
        dir_list.append(dirs)

    for weights in list_weights:
        if wfb:
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        else:
            # flatten
            boxes = [box for boxes in boxes_list for box in boxes]
            scores = [score * weight for weight, scores in zip(weights, scores_list) for score in scores]
            labels = [label for labels in labels_list for label in labels]
            dirs = [dir_ for dirs in dir_list for dir_ in dirs]

        box_infos = []
        for box, score, label, dir_ in zip(boxes, scores, labels, dirs):
            x1, y1, x2, y2 = box
            box_infos.append([label, x1, y1, x2, y2, score, dir_])
        digit_sum, final_box_infos = filter_overlap(box_infos)

        # digit_sum = sum(list(map(int, labels)))
        # data_lines.append([file_name[:-4], digit_sum])
        weight_name = '_'.join([str(e) for e in weights])
        data_lines_dict[weight_name].append([file_name[:-4], digit_sum])
        valid_box_infos[weight_name][file_name[:-4]] = final_box_infos

    return data_lines_dict, valid_box_infos


def merge_results(list_input_dir,
                  save_dir='./submission_wfb',
                  list_weights=None,
                  list_conf_thresh=None,
                  wfb=True,
                  save_txt_dir=None):
    if list_conf_thresh is None:
        list_conf_thresh = [0.0] * len(list_input_dir)
    if list_weights is None:
        list_weights = [[1] * len(list_input_dir)]

    # Get all file_name
    list_file_name = []
    for input_dir in list_input_dir:
        list_file_name += [p.name for p in input_dir.glob('*.txt')]
    list_file_name = list(set(list_file_name))

    if isinstance(save_txt_dir, (str, Path)):
        save_txt_dir = Path(save_txt_dir)

    header = ['id', 'digit_sum']
    data_lines_dict = {'_'.join([str(e) for e in weights]): [] for weights in list_weights}
    valid_box_infos = {'_'.join([str(e) for e in weights]): {} for weights in list_weights}

    # with Manager() as manager:
    #     data_lines_dict = manager.dict()
    #     valid_box_infos = manager.dict()
    #     for weights in list_weights:
    #         weight_name = '_'.join([str(e) for e in weights])
    #         data_lines_dict[weight_name] = []
    #         valid_box_infos[weight_name] = {}
    #     with Pool(8) as pool:
    #         results = [pool.apply_async(get_acc_by_file, args=(data_lines_dict,
    #                                                            valid_box_infos,
    #                                                            file_name,
    #                                                            list_input_dir,
    #                                                            list_weights,
    #                                                            list_conf_thresh,
    #                                                            wfb,))
    #                    for file_name in tqdm(list_file_name)]
    #         results = [ret.get() for ret in results]
    # print(data_lines_dict)
    with Pool(8) as pool:
        results = [pool.apply_async(get_acc_by_file, args=(file_name,
                                                           list_input_dir,
                                                           list_weights,
                                                           list_conf_thresh,
                                                           wfb,))
                   for file_name in list_file_name]
        results = [tuple(ret.get()) for ret in tqdm(results)]
    for sub_data_lines, sub_valid_box_info in results:
        for k, v in sub_data_lines.items():
            data_lines_dict[k] += v
        for weight_name, dict_val in sub_valid_box_info.items():
            for file_name, val in dict_val.items():
                valid_box_infos[weight_name][file_name] = val
    # print(sub_valid_box_info)
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)
    # save all csv file
    for weights in list_weights:
        data_lines_name = '_'.join([str(e) for e in weights])
        save_csv_path = save_dir.joinpath('submission_wfb_' + data_lines_name + '.csv')
        with open(str(save_csv_path), 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # write multiple rows
            writer.writerows(data_lines_dict[data_lines_name])
    # save all txt_file
    if save_txt_dir is not None:
        for weights in list_weights:
            data_lines_name = '_'.join([str(e) for e in weights])
            save_txt_dir = save_txt_dir.joinpath(data_lines_name)
            if not save_txt_dir.is_dir():
                save_txt_dir.mkdir(parents=True)
            for file_name, box_infos in valid_box_infos[data_lines_name].items():
                save_txt_path = save_txt_dir.joinpath(file_name + '.txt')
                lines = []
                for info in box_infos:
                    cls, x_min, y_min, x_max, y_max, score, _ = info
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    w = x_max - x_min
                    h = y_max - y_min
                    lines.append(' '.join(list(map(str, [cls, x_center, y_center, w, h, score]))))
                with open(str(save_txt_path), 'w') as f:
                    f.write('\n'.join(lines))


if __name__ == '__main__':

    merge_results(
        list_input_dir=[
            Path('./txt_predicts/yolov5/220327_normal_9split_1024'),
            Path('./txt_predicts/yolov5/220327_normal_16split_1024'),
            Path('./txt_predicts/yolov5/220324_tiny_25split_1024'),
            Path('./txt_predicts/yolov5/220414_fold0_e50_1024'),
            Path('./txt_predicts/yolov5/220414_fold0_e50_768'),
            Path('./txt_predicts/yolov5/220414_fold0_e50_1280'),
            Path('./txt_predicts/yolov5/220414_fold0_e50_1536'),
        ],
        save_dir='./220414_7merge',
        list_weights=[[1, 1, 1, 1, 1, 1, 1]],
        list_conf_thresh=[0.8, 0.8, 0.7, 0.9, 0.95, 0.9, 0.95],
        wfb=False
    )
