# copied and modified from https://github.com/ultralytics/yolov5/pull/668
import os
import torch
import numpy as np
import cv2
import tqdm
import time
import itertools
import math
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, clip_coords
from utils.torch_utils import select_device
from utils.plots import Annotator, save_one_box


class Yolov5():
    def __init__(self,
                 weights_path,
                 device='',
                 img_size=640,
                 conf_thres=0.4,
                 iou_thres=0.5,
                 augment=False,
                 agnostic_nms=False,
                 classes=None):
        self.device = select_device(device)
        self.weights_name = os.path.split(weights_path)[-1]
        self.model = attempt_load(weights_path, map_location=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = check_img_size(img_size, s=self.model.stride.max())
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
        if self.device.type != 'cpu':
            self.burn()

    def __str__(self):
        out = ['Model: %s' % self.weights_name]
        out.append('Image size: %s' % self.imgsz)
        out.append('Confidence threshold: %s' % self.conf_thres)
        out.append('IoU threshold: %s' % self.iou_thres)
        out.append('Augment: %s' % self.augment)
        out.append('Agnostic nms: %s' % self.agnostic_nms)
        if self.classes is not None:
            filter_classes = [self.names[each_class] for each_class in self.classes]
            out.append('Classes filter: %s' % filter_classes)
        out.append('Classes: %s' % self.names)

        return '\n'.join(out)

    def burn(self):
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img)  # run once

    def predict(self, img0, visualize_path=None):
        img = letterbox(img0, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)  # uint8 to float32

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=self.augment)[0]
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres,
            classes=self.classes, agnostic=self.agnostic_nms)

        det = pred[0]

        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        if isinstance(visualize_path, str):
            annotator = Annotator(img0, line_width=3, example=str(self.model.names))
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (self.names[int(cls)], conf)
                annotator.box_label(xyxy, label, color=self.colors[int(cls)])

            cv2.imwrite(visualize_path, annotator.result())

        min_max_list = self.min_max_list(det, output_size=(self.imgsz, self.imgsz))
        return [min_max_list]

    def predict_batch(self, img0s, list_visualize_path=None, return_pred=False):
        imgs = []
        for img0 in img0s:
            img = letterbox(img0, new_shape=self.imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)  # uint8 to float32
            imgs.append(img)

        imgs = np.array(imgs)
        imgs = torch.from_numpy(imgs).to(self.device)
        imgs = imgs.half() if self.half else imgs.float()
        imgs /= 255.0  # 0 - 255 to 0.0 - 1.0

        with torch.no_grad():
            # Run model
            inf_out, _ = self.model(imgs, augment=self.augment)  # inference and training outputs

            # Run NMS
            preds = non_max_suppression(
                inf_out,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres)
        # scale boxes
        for idx in range(len(preds)):
            if preds[idx] is not None and len(preds[idx]):
                preds[idx][:, :4] = scale_coords(
                    imgs[idx].shape[1:],
                    preds[idx][:, :4],
                    img0s[idx].shape[:2]).round()

        if list_visualize_path is not None:
            for save_path, det, img, img0 in zip(list_visualize_path, preds, imgs, img0s):
                annotator = Annotator(img0, line_width=3, example=str(self.model.names))
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    annotator.box_label(xyxy, label, color=self.colors[int(cls)])
                cv2.imwrite(save_path, annotator.result())

        batch_output = []
        # print('self.imgsz', self.imgsz)
        for idx, pred in enumerate(preds):
            batch_output.append(self.min_max_list(pred, output_size=img0s[idx].shape[:2]))
        if return_pred:
            return batch_output, preds
        return batch_output

    def predict_overlap_split(
        self, img0,
        first_resize=(3024, 3024), split_size=(1024, 1024),
        visualize_path=None
    ):
        # split_image into multiple of sub-image
        img0 = cv2.resize(img0, first_resize)
        patches, (width, height) = self.overlap_split(img0, split_size)
        imgs = [patch[0] for patch in patches]
        batch_output, preds = self.predict_batch(imgs, return_pred=True)
        # remap coordinate
        for idx in range(len(preds)):
            if preds[idx] is not None and len(preds[idx]):
                start_w, stop_w, start_h, stop_h = patches[idx][1]
                preds[idx][:, [0, 2]] += start_w  # x padding
                preds[idx][:, [1, 3]] += start_h  # y padding
                clip_coords(preds[idx][:, :4], img0.shape[:2])
        # preds = non_max_suppression(
        #     preds, self.conf_thres, self.iou_thres,
        #     classes=self.classes, agnostic=self.agnostic_nms)

        det = torch.cat(preds, dim=0)
        if isinstance(visualize_path, str):
            annotator = Annotator(img0, line_width=3, example=str(self.model.names))
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (self.names[int(cls)], conf)
                annotator.box_label(xyxy, label, color=self.colors[int(cls)])

            cv2.imwrite(visualize_path, annotator.result())

        min_max_list = self.min_max_list(det, output_size=first_resize)
        return [min_max_list]

    def min_max_list(self, det, output_size):
        width_out, height_out = output_size
        min_max_list = []
        for i, c in enumerate(det[:, -1]):
            x_min = min(int(det[i][0]), int(det[i][2]))
            x_max = max(int(det[i][0]), int(det[i][2]))
            y_min = min(int(det[i][1]), int(det[i][3]))
            y_max = max(int(det[i][1]), int(det[i][3]))
            w = x_max - x_min
            h = y_max - y_min
            x_center = (x_min + 0.5 * w) / width_out
            y_center = (y_min + 0.5 * h) / height_out
            w = w / width_out
            h = h / height_out
            obj = {
                'bndbox': {
                    'x_center': x_center,
                    'y_center': y_center,
                    'w': w,
                    'h': h
                },
                'name': self.names[int(c)],
                'conf': float(det[i][4]),
                'color': self.colors[int(det[i][5])]
            }
            min_max_list.append(obj)

        return min_max_list

    def overlap_split(self, image, split_size):
        # split image into sub_image with split_size
        height, width = image.shape[:2]
        split_size_w, split_size_h = split_size
        n_w, n_h = width / split_size_w, height / split_size_h

        if n_w != 1:
            n_w = n_w + 1
        if n_h != 1:
            n_h = n_h + 1

        n_w, n_h = int(n_w), int(n_h)

        overlap_w = round((n_w * split_size_w - width) / (n_w - 1)) if n_w > 1 else 0
        overlap_h = round((n_h * split_size_h - height) / (n_h - 1)) if n_h > 1 else 0

        patches = []
        for i, j in itertools.product(range(n_w), range(n_h)):
            if i + 1 == n_w:
                start_w = max(width - split_size_w, 0)
            else:
                start_w = i * (split_size_w - overlap_w)

            if i + 1 == n_w:
                stop_w = width
            else:
                stop_w = start_w + split_size_w

            if j + 1 == n_h:
                start_h = max(height - split_size_h, 0)
            else:
                start_h = j * (split_size_h - overlap_h)

            if j + 1 == n_h:
                stop_h = height
            else:
                stop_h = start_h + split_size_h

            patches.append((image[start_h:stop_h, start_w:stop_w], (start_w, stop_w, start_h, stop_h)))
        return patches, (width, height)


def chunks(lst, bs=1):
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]


def predict_and_save_txt(
    weights_path,
    save_dir,
    input_dir='../dataset/test',
    device='',
    img_size_model=1024,
    conf_thres=0.4,
    iou_thres=0.45,
    batch_size=2,
    infer_mode='batch',
    first_resize=(3024, 3024),
    split_size=(1024, 1024),
    invert=False
):
    input_dir = Path(input_dir)
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)
    # get list of file_path
    list_image_inputs = list(input_dir.glob('*'))
    model = Yolov5(
        weights_path=weights_path,
        device=device,
        img_size=img_size_model,
        conf_thres=conf_thres,
        iou_thres=iou_thres)
    if infer_mode == 'overlap_split':
        batch_size = 1
    max_num_batch = len(list_image_inputs) // batch_size
    for idx, image_paths in enumerate(chunks(list_image_inputs, bs=batch_size)):
        start_time = time.time()
        list_img = []
        for image_path in image_paths:
            # list_img.append(cv2.imread(str(image_path)))
            image = cv2.imread(str(image_path))
            if invert:
                image = ~image
            list_img.append(image)
        if infer_mode == 'batch':
            preds = model.predict_batch(list_img)
        elif infer_mode == 'overlap_split':
            preds = model.predict_overlap_split(
                list_img[0], first_resize=first_resize, split_size=split_size)

        for image_path, pred in zip(image_paths, preds):
            txt_save_path = save_dir.joinpath(image_path.name).with_suffix('.txt')
            if txt_save_path.is_file():
                continue
            lines = []
            for obj in pred:
                cls_idx = int(obj['name'])
                conf = float(obj['conf'])
                x_center = float(obj['bndbox']['x_center'])
                y_center = float(obj['bndbox']['y_center'])
                w = float(obj['bndbox']['w'])
                h = float(obj['bndbox']['h'])
                lines.append(' '.join(list(map(str, [cls_idx, x_center, y_center, w, h, conf]))))
            with open(str(txt_save_path), 'w') as f:
                f.write('\n'.join(lines))
        print('process ok at : {}/{}'.format(idx, max_num_batch), time.time() - start_time)


if __name__ == '__main__':
    predict_and_save_txt(
        weights_path='./runs/train/yolov5x_220414_normal/best.pt',
        input_dir='../dataset/test',
        save_dir='../txt_predicts/yolov5/220414_fold0_e50_1536',
        img_size_model=1536,
        conf_thres=0.25,
        iou_thres=0.45,
        batch_size=8,
        invert=False
    )

    predict_and_save_txt(
        weights_path='./runs/train/yolov5x_220414_normal/best.pt',
        input_dir='../dataset/test',
        save_dir='../txt_predicts/yolov5/220414_fold0_e50_1280',
        img_size_model=1280,
        conf_thres=0.25,
        iou_thres=0.45,
        batch_size=8,
        invert=False
    )

    predict_and_save_txt(
        weights_path='./runs/train/yolov5x_220414_normal/best.pt',
        input_dir='../dataset/test',
        save_dir='../txt_predicts/yolov5/220414_fold0_e50_1024',
        img_size_model=1024,
        conf_thres=0.25,
        iou_thres=0.45,
        batch_size=8,
        invert=False
    )

    predict_and_save_txt(
        weights_path='./runs/train/yolov5x_220414_normal/best.pt',
        input_dir='../dataset/test',
        save_dir='../txt_predicts/yolov5/220414_fold0_e50_768',
        img_size_model=768,
        conf_thres=0.25,
        iou_thres=0.45,
        batch_size=8,
        invert=False
    )

    predict_and_save_txt(
        weights_path='./runs/train/yolov5x_220327_normal/best.pt',
        input_dir='../dataset/test',
        save_dir='../txt_predicts/yolov5/220327_normal_9split_1024',
        img_size_model=1024,
        conf_thres=0.25,
        iou_thres=0.45,
        batch_size=1,
        infer_mode='overlap_split',
        first_resize=(3024, 3024),
        split_size=(1024, 1024)
    )

    predict_and_save_txt(
        weights_path='./runs/train/yolov5x_220327_normal/best.pt',
        input_dir='../dataset/test',
        save_dir='../txt_predicts/yolov5/220327_normal_16split_1024',
        img_size_model=1024,
        conf_thres=0.25,
        iou_thres=0.45,
        batch_size=1,
        infer_mode='overlap_split',
        first_resize=(4024, 4024),
        split_size=(1024, 1024)
    )

    predict_and_save_txt(
        weights_path='./runs/train/yolov5x_220324_tiny/best.pt',
        input_dir='../dataset/test',
        save_dir='../txt_predicts/yolov5/220324_tiny_25split_1024',
        img_size_model=1024,
        conf_thres=0.25,
        iou_thres=0.45,
        batch_size=1,
        infer_mode='overlap_split',
        first_resize=(5024, 5024),
        split_size=(1024, 1024)
    )
