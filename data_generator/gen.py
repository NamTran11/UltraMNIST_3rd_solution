import json
import sys
import yaml
import cv2
import numpy as np
import imgaug.augmenters as iaa
import pandas as pd
import datetime
import re
from typing import List, Tuple
from pathlib import Path


class ImageGenerator:
    def __init__(self, config):
        self.data_path = config.get('data_path', './digit-recognizer/train.csv')
        self.save_dir = Path(config['save_dir'])
        self.num_sample = config['num_sample']
        self.save_mask = config.get('save_mask', True)
        self.output_size = config.get('output_size', (4000, 4000))
        self.circle_config = config['circle_config']
        self.triangle_config = config['triangle_config']
        self.grid_config = config['grid_config']
        self.digit_config = config['digit_config']
        self.resize_output = config.get('resize_output', None)
        self.yoloformat = config.get('yoloformat', False)
        self.export_txt_name = config.get('export_txt_name', True)
        self.tiny_digit_config = config.get('tiny_digit_config', None)

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

        train = pd.read_csv(self.data_path)
        if 'test.csv' in self.data_path:
            self.train_imgs = train.iloc[:, :].copy()
            label_path = self.data_path.replace('test.csv', 'test_label.csv')
            self.train_labels = pd.read_csv(label_path)['Label'].to_numpy()
        else:
            self.train_imgs = train.iloc[:, 1:].copy()
            self.train_labels = train.iloc[:, 0].to_numpy()
        self.train_imgs = self.train_imgs.to_numpy().reshape(len(self.train_imgs), 28, 28)
        self.train_idxs = list(range(0, len(self.train_imgs)))

    def gen_test(self):
        bg = self.get_grid()
        crl_img = self.get_circle()
        trg_img = self.get_triangle()
        bg = self.xor_image(crl_img, bg)
        bg = self.xor_image(trg_img, bg)
        gt_img, json_label = self.get_digit_mask()
        bg = cv2.bitwise_xor(gt_img, bg)
        return bg

    def gen(self):
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

        p = self.digit_config.get('size_p', None)
        if isinstance(p, str):
            p = np.load(p)

        list_file_name = []
        for img_idx in range(self.num_sample):
            # get grid
            background = self.get_grid()
            # generate circle
            if np.random.rand() <= self.circle_config['prob_circle']:
                num_triangle = np.random.randint(
                    self.circle_config['min_circle'],
                    self.circle_config['max_circle'] + 1)
                for _ in range(num_triangle):
                    circle_img = self.get_circle(
                        min_r=self.circle_config['min_circle_radius'],
                        max_r=self.circle_config['max_circle_radius'],
                    )
                    background = self.xor_image(circle_img, background)

            # generate triangle
            if np.random.rand() <= self.triangle_config['prob_triangle']:
                num_triangle = np.random.randint(
                    self.triangle_config['min_triangle'],
                    self.triangle_config['max_triangle'] + 1)
                for _ in range(num_triangle):
                    trg_img = self.get_triangle(
                        min_size=self.triangle_config['min_triangle_size'],
                        max_size=self.triangle_config['max_triangle_size'],
                    )
                    background = self.xor_image(trg_img, background)
            try:
                # get mask digit
                digit_mask, json_label, tiny_boxs, tiny_labels = self.get_digit_mask(
                    min_num_digit=self.digit_config['min_num_digit'],
                    max_num_digit=self.digit_config['max_num_digit'],
                    min_size_digit=self.digit_config['min_size_digit'],
                    max_size_digit=self.digit_config['max_size_digit'],
                    p=p,
                    size=self.output_size,
                    tiny_config=self.tiny_digit_config,
                )
                background = cv2.bitwise_xor(digit_mask, background)

                # save all result
                time_post_fix = re.sub(r"[^a-zA-Z0-9]", "", str(datetime.datetime.now()))
                img_name = str(img_idx) + '_' + time_post_fix + '.jpg'
                save_img_path = self.save_dir.joinpath(img_name)
                save_json_path = save_img_path.with_suffix('.json')

                if self.tiny_digit_config is not None:
                    glob_w, glob_h = self.output_size
                    out_w, out_h = self.tiny_digit_config['output_size']

                    # pick location to crop
                    cdd_box = tiny_boxs[0]
                    x_min_c, y_min_c, x_max_c, y_max_c = cdd_box
                    post_x = np.random.randint(max(0, x_max_c - out_w), min(x_min_c, glob_w - out_w))
                    post_y = np.random.randint(max(0, y_max_c - out_h), min(y_min_c, glob_h - out_h))

                    cut_box = [post_x, post_y, post_x + out_w, post_y + out_h]
                    background = background[post_y: post_y + out_h, post_x: post_x + out_w]
                    cv2.imwrite(str(save_img_path), background)
                    if self.save_mask:
                        digit_mask = digit_mask[post_y: post_y + out_h, post_x: post_x + out_w]
                        save_mask_path = save_img_path.with_suffix('.png')
                        cv2.imwrite(str(save_mask_path), digit_mask)
                    # check is inner and save as yoloformat
                    lines = []
                    for obj_info in json_label['shapes']:
                        is_inner, new_points = is_inner_cut_box(
                            cut_box,
                            obj_info['points'],
                            overlap_ratio=0.7,
                            size_thresh=self.tiny_digit_config.get('thresh_large_size', 35))
                        if is_inner:
                            gt = int(obj_info['label'])
                            box_info = list(map(str, get_yolo_rect(new_points, self.tiny_digit_config['output_size'])))
                            line = ' '.join([str(gt)] + box_info)
                            lines.append(line)
                    save_txt_path = save_json_path.with_suffix('.txt')
                    with open(str(save_txt_path), 'w') as f:
                        f.write('\n'.join(lines))

                else:
                    if self.yoloformat and self.resize_output is not None:
                        background = cv2.resize(background, tuple(self.resize_output))
                        if self.save_mask:
                            digit_mask = cv2.resize(digit_mask, tuple(self.resize_output))
                    cv2.imwrite(str(save_img_path), background)
                    if self.save_mask:
                        save_mask_path = save_img_path.with_suffix('.png')
                        cv2.imwrite(str(save_mask_path), digit_mask)

                    if self.yoloformat:
                        # save as yoloformat: class, x_center, y_center, w, h
                        lines = []
                        for obj_info in json_label['shapes']:
                            gt = int(obj_info['label'])
                            box_info = list(map(str, get_yolo_rect(obj_info['points'], self.output_size)))
                            line = ' '.join([str(gt)] + box_info)
                            lines.append(line)
                        save_txt_path = save_json_path.with_suffix('.txt')
                        with open(str(save_txt_path), 'w') as f:
                            f.write('\n'.join(lines))

                    if self.resize_output is None:
                        json_label['imagePath'] = save_img_path.name
                        with open(str(save_json_path), 'w') as f:
                            json.dump(json_label, f, indent=4, ensure_ascii=False)
                list_file_name.append(save_json_path.stem)
                print('process ok at {}/{}'.format(img_idx, self.num_sample))
            except:
                print('__error at image__', img_idx)
                continue
        if self.export_txt_name:
            export_txt_path = str(self.save_dir) + '.txt'
            with open(export_txt_path, 'w') as f:
                f.write('\n'.join(list_file_name))

    def xor_image(self, img: np.ndarray, bg: np.ndarray):
        """`Insert` and image into background with XOR operator

        Args:
            img (np.ndarray): _description_
            bg (np.ndarray): _description_

        Returns:
            bg: final result
        """
        h_bg, w_bg = bg.shape[:2]
        h_img, w_img = img.shape[:2]
        y_post = np.random.randint(0, h_bg - h_img - 2)
        x_post = np.random.randint(0, w_bg - w_img - 2)
        # img_crop = bg[y_post: y_post + h_img, x_post: x_post + w_img]
        bg[y_post: y_post + h_img, x_post: x_post + w_img] = cv2.bitwise_xor(
            bg[y_post: y_post + h_img, x_post: x_post + w_img],
            img)
        return bg

    def get_digit_mask(self,
                       min_num_digit: int = 2,
                       max_num_digit: int = 5,
                       min_size_digit: int = 100,
                       max_size_digit: int = 1000,
                       p=None,
                       tiny_config=None,
                       size: Tuple[int, int] = (4000, 4000)) -> Tuple[np.ndarray, dict]:
        """from MNIST dataset, generate mask image

        Args:
            min_num_digit (int, optional): min number of digit. Defaults to 2.
            max_num_digit (int, optional): max number of digit. Defaults to 5.
            min_size_digit (int, optional): min size of digit. Defaults to 100.
            max_size_digit (int, optional): max size of digit. Defaults to 1000.
            num_digit (int, optional): number of digit. Defaults to 5.
            size (Tuple[int, int], optional): size of output mask. Defaults to (4000, 4000).

        Returns:
            ret_img (np.ndarray): final mask
            json_object (dict): final result as json object for saving
        """
        def is_overlap(box, boxes):
            for valid_box in boxes:
                x_min, y_min, x_max, y_max = valid_box
                x1, y1, x2, y2 = box
                if max(x_min, x1) < min(x_max, x2) and max(y_min, y1) < min(y_max, y2):
                    return True
            return False

        num_digit = np.random.randint(min_num_digit, max_num_digit + 1)
        h, w = size
        list_idx = np.random.choice(self.train_idxs, size=num_digit, replace=False)
        ret_img = np.zeros(size, dtype=np.uint8)
        list_img = [self.train_imgs[idx] for idx in list_idx]
        list_label = [self.train_labels[idx] for idx in list_idx]
        list_size = []
        tiny_boxs = []
        tiny_labels = []

        if self.tiny_digit_config is not None:
            # set size for each of digit
            min_tiny_size, max_tiny_size = self.tiny_digit_config['tiny_size']
            min_medium_size, max_medium_size = self.tiny_digit_config['medium_size']
            min_large_size, max_large_size = self.tiny_digit_config['large_size']
            if num_digit == 3:
                # 2 tiny, 1 large
                list_size = [
                    np.random.randint(min_tiny_size, max_tiny_size),
                    np.random.randint(min_tiny_size, max_tiny_size),
                    np.random.randint(min_large_size, max_large_size),
                ]

            if num_digit == 4:
                # 2 tiny, 1 medium, 1 large
                list_size = [
                    np.random.randint(min_tiny_size, max_tiny_size),
                    np.random.randint(min_tiny_size, max_tiny_size),
                    np.random.randint(min_medium_size, max_medium_size),
                    np.random.randint(min_large_size, max_large_size),
                ]
            if num_digit == 5:
                # 2 tiny 1 medium, 2 large
                list_size = [
                    np.random.randint(min_tiny_size, max_tiny_size),
                    np.random.randint(min_tiny_size, max_tiny_size),
                    np.random.randint(min_medium_size, max_medium_size),
                    np.random.randint(min_large_size, max_large_size),
                    np.random.randint(min_large_size, max_large_size),
                ]

        for idx in range(len(list_img)):
            # thresholding
            list_img[idx] = np.array(list_img[idx], dtype=np.uint16)
            _, thresh = cv2.threshold(list_img[idx], 0, 255, cv2.THRESH_OTSU)

            # resize digit
            if len(list_size) != 0:
                new_size = list_size[idx]
            else:
                if p is not None:
                    # version 220320 with train
                    # new_size = np.random.choice(list(range(23, 1598)), size=1, replace=True, p=p)[0]
                    # version 220320 with test
                    new_size = np.random.choice(list(range(28, 1603)), size=1, replace=True, p=p)[0]
                    # version 220321
                    # new_size = np.random.choice(list(range(8, 1583)), size=1, replace=True, p=p)[0]
                else:
                    new_size = np.random.randint(min_size_digit, max_size_digit)

            list_img[idx] = cv2.resize(thresh, (new_size, new_size), interpolation=cv2.INTER_NEAREST)
            # get exactly boundary of digit
            v_prj = np.sum(list_img[idx], axis=1)
            h_prj = np.sum(list_img[idx], axis=0)
            y_top, y_bot = self.find_boundary(v_prj, 1)
            x_left, x_right = self.find_boundary(h_prj, 1)
            assert -1 not in [y_top, y_bot, x_left, x_right], "__[get_digit_mask]__: invalid image"
            list_img[idx] = list_img[idx][y_top: y_bot, x_left: x_right]
            # return list_img[idx] * 255

        # sort list box by size
        order_idxs = np.argsort([max(list(img.shape[:2])) for img in list_img])
        order_idxs = list(order_idxs)[::-1]  # reverse

        list_box = []
        list_gt = []
        for idx in order_idxs:
            img_digit, label = list_img[idx], list_label[idx]
            for _ in range(0, 100):
                h_d, w_d = img_digit.shape[:2]
                y_post = np.random.randint(0, h - h_d - 2)
                x_post = np.random.randint(0, w - w_d - 2)
                box = [x_post, y_post, x_post + w_d, y_post + h_d]
                # make sure all digits do not overlap with each other
                if not is_overlap(box, list_box):
                    ret_img[y_post: y_post + h_d, x_post: x_post + w_d] = img_digit
                    list_box.append(box)
                    list_gt.append(label)
                    break
        if self.tiny_digit_config is not None:
            tiny_boxs = list_box[-2:]
            tiny_labels = list_gt[-2:]
        # convert label as json_object
        json_label = {
            "version": "4.5.7",
            "flags": {},
            "imageData": None,
            "shapes": [],
            "tables": [],
            "imagePath": "",
            "imageHeight": h,
            "imageWidth": w,
        }
        for box, gt_label in zip(list_box, list_gt):
            x_min, y_min, x_max, y_max = list(map(int, box))
            json_label["shapes"].append({
                "label": str(gt_label),
                "points": [[x_min, y_min], [x_max, y_max]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            })
        return ret_img, json_label, tiny_boxs, tiny_labels

    def get_circle(self, min_r: int = 500, max_r: int = 1500) -> np.ndarray:
        """Generate the circle image

        Args:
            min_r (int, optional): min radius. Defaults to 500.
            max_r (int, optional): max radius. Defaults to 1500.

        Returns:
            np.ndarray: final result
        """
        r = np.random.randint(min_r, max_r)
        h, w = r * 2, r * 2
        img = np.zeros((h, w), dtype=np.uint8)
        img = cv2.circle(img, (r, r), r - 1, 255, -1)
        return img

    def get_triangle(self, min_size: int = 2000, max_size: int = 3000) -> np.ndarray:
        """Generate triangle image

        Args:
            min_size (int, optional): min size of square which contains triangle. Defaults to 2000.
            max_size (int, optional): max size of square. Defaults to 3000.

        Returns:
            img (np.ndarray): final result
        """
        def is_triangle(x1, y1, x2, y2, x3, y3):
            a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            return False if a == 0 else True
        size = np.random.randint(min_size, max_size)
        h, w = size, size
        img = np.zeros((h, w), dtype=np.uint8)
        while True:
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            x2 = np.random.randint(0, w)
            y2 = np.random.randint(0, h)
            x3 = np.random.randint(0, w)
            y3 = np.random.randint(0, h)
            if is_triangle(x1, y1, x2, y2, x3, y3):
                x_min = min(x1, x2, x3)
                x_max = max(x1, x2, x3)
                y_min = min(y1, y2, y3)
                y_max = max(y1, y2, y3)
                pts = np.int32([[x1, y1], [x2, y2], [x3, y3]])
                cv2.drawContours(img, [pts], -1, 255, -1)
                return img[y_min: y_max, x_min: x_max]

    def get_grid(self,
                 min_cell: int = 20,
                 max_cell: int = 50,
                 output_size: int = 4000,
                 prob: float = 0.5) -> np.ndarray:
        """Generate grid of black/white squares

        Args:
            min_cell (int, optional): the minimum number of cell in grid. Defaults to 20.
            max_cell (int, optional): the maximum number of cell in grid. Defaults to 50.
            output_size (int, optional): _description_. Defaults to 4000.
            prob (float, optional): probabylity to generate a white square. Defaults to 0.5.

        Returns:
            np.ndarray: final result
        """

        num_cells = np.random.randint(min_cell, max_cell)
        size = num_cells * 100
        crop_size = 1000

        height, width = size, size
        assert height % num_cells == 0 and width % num_cells == 0, \
            "__[ImageGenerator: get_grid]__: invalid info"
        img = np.zeros((height, width), dtype=np.uint8)
        num_block_vtc = height // num_cells
        num_block_hrz = width // num_cells
        for i_v in range(num_cells):
            for i_h in range(num_cells):
                if np.random.rand() > prob:
                    x_min = i_h * num_block_hrz
                    y_min = i_v * num_block_vtc
                    x_max = x_min + num_block_hrz
                    y_max = y_min + num_block_vtc
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), 255, -1)
        # transform image
        # img =  Transform.rotate3D(img, theta=50, phi=0, gamma=0, dx=0, dy=0, dz=0)
        seq = iaa.Sequential([
            iaa.PerspectiveTransform(scale=(0.01, 0.3)),
            iaa.Affine(shear=(-20, 20)),
            iaa.CropToFixedSize(width=crop_size, height=crop_size)
            # iaa.Crop(px=(0, 10))
        ])
        img = seq(image=img)
        img = cv2.resize(img, (output_size, output_size))
        return img

    def find_boundary(self, prj: List[int], thresh: int) -> Tuple[int, int]:
        """get the first and last index in a list whose greater value than thresh

        Args:
            prj (List[int]): input list
            thresh (int): threshold

        Returns:
            Tuple[int, int]: min, and max index
        """
        p_min = -1
        p_max = -1
        for i in range(0, len(prj)):
            if prj[i] >= thresh and p_min == -1:
                p_min = i
            if prj[len(prj) - 1 - i] >= thresh and p_max == -1:
                p_max = len(prj) - i
            if -1 not in [p_min, p_max]:
                break
        if -1 in [p_min, p_max] or abs(p_max - p_min) < 2 or p_max < p_min:
            return -1, -1
        else:
            return p_min, p_max


if __name__ == '__main__':
    global_config_path = sys.argv[1]
    with open(str(global_config_path)) as f:
        configs = yaml.safe_load(f)
    image_generator = ImageGenerator(configs)
    image_generator.gen()
