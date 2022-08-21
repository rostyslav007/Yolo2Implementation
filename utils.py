import numpy as np
import torch
from data_aug.data_aug import *
import torch.nn as nn
from config import device
from PIL import Image, ImageOps
from data_aug.bbox_util import *
from collections import Counter


def load_anchors(anchor_path):
    anch = []

    with open(anchor_path, 'r') as file:
        for line in file.readlines():
            h, w = line.split()
            anch.append([0, 0, float(h), float(w)])
    return torch.tensor(anch, dtype=torch.float32)


def iou(boxA, boxB):
    device = boxA.device
    x1, y1, w1, h1 = boxA[:, 0], boxA[:, 1], boxA[:, 2], boxA[:, 3]
    x2, y2, w2, h2 = boxB[:, 0], boxB[:, 1], boxB[:, 2], boxB[:, 3]

    x_a = torch.max(x1 - w1 / 2, x2 - w2 / 2)
    y_a = torch.max(y1 - h1 / 2, y2 - h2 / 2)
    x_b = torch.min(x1 + w1 / 2, x2 + w2 / 2)
    y_b = torch.min(y1 + h1 / 2, y2 + h2 / 2)

    intersection = torch.abs(torch.max(x_b - x_a, torch.zeros(1).to(device)) * \
                             torch.max(y_b - y_a, torch.zeros(1).to(device)))
    area_a = torch.abs(w1 * h1)
    area_b = torch.abs(w2 * h2)
    union = area_a + area_b - intersection

    return intersection / union


def generate_images(path, label_path, img_shape):
    img_size = img_shape[0]
    img = cv2.resize(cv2.imread(path), img_shape)[:, :, ::-1]
    boxes = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            c, x, y, w, h = [float(n) for n in line.split()]
            boxes.append(
                [(x - w / 2) * img_size, (y - h / 2) * img_size, (x + w / 2) * img_size, (y + h / 2) * img_size, c])
    bboxes = np.array(boxes)
    img_aug, boxes_aug = augment_image(img, bboxes)

    string = 'abcdefghiklmnopqrsuvwxyz0123456789'
    part = ''.join(random.sample(string, 4))
    grey_image = ImageOps.grayscale(Image.fromarray(img_aug))

    path_list = label_path.split('\\')
    path_list[1] = 'augmented_samples'
    name = path_list[-1].split('.')[0]

    img_p = os.path.join(path_list[0], path_list[1], 'images', 'train', name + part + '.jpg')
    file_path = os.path.join(path_list[0], path_list[1], 'labels', 'train', name + part + '.txt')

    with open(file_path, 'w') as file:
        for box in boxes_aug:
            x1, y1, x2, y2, c = float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(box[4])
            x_c, y_c = (x1 + x2) / 2 / img_size, (y1 + y2) / 2 / img_size
            w, h = (x2 - x1) / img_size, (y2 - y1) / img_size
            file.write(str(c) + ' ' + str(x_c) + ' ' + str(y_c) + ' ' + str(w) + ' ' + str(h) + '\n')
    grey_image.save(img_p)
    return 'Generated'


def augment_image(img, bboxes, proba=0.2, scale=0.3, translate=0.2, rotate_angle=180):
    seq = Sequence([RandomRotate(rotate_angle),
                    RandomHorizontalFlip(proba)])

    img_, bboxes_ = seq(img.copy(), bboxes.copy())

    return img_, bboxes_


def non_max_suppression(imgs, predictions, anchors, confidence_threshold=0.5, iou_threshold=0.5):
    num_anchors = anchors.shape[0]
    sigmoid = nn.Sigmoid()
    p_b, p_v, p_s, p_s = predictions.shape
    predictions = predictions.reshape(p_b, num_anchors, -1, p_s, p_s)
    predictions[:, :, 0, :, :] = sigmoid(predictions[:, :, 0, :, :])
    predictions[:, :, 3:5, :, :] = p_s * anchors[:, 2:].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * \
                                   torch.exp(predictions[:, :, 3:5, :, :])
    predictions[:, :, 1:3, :, :] = sigmoid(predictions[:, :, 1:3, :, :])
    predictions[:, :, 5, :, :] = sigmoid(predictions[:, :, 5, :, :])
    # B S S An 6
    predictions = predictions.permute(0, 3, 4, 1, 2)

    predictions = predictions.view(p_b, p_s, p_s, p_v)

    relevant_boxes_indices = torch.nonzero(predictions[:, :, :, ::6] > confidence_threshold)

    bboxes = []
    for ind in relevant_boxes_indices:
        b, s1, s2, a = [int(i) for i in ind]
        bboxes.append(predictions[b, s1, s2, a*6:(a+1)*6])
        bboxes[-1][1] += s1
        bboxes[-1][2] += s2
    bboxes = sorted(bboxes, key=lambda t: t[0])[::-1]
    cleaned_bboxes = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes[1:]
                  if iou(box[1:5].unsqueeze(0), chosen_box[1:5].unsqueeze(0)) < iou_threshold]
        cleaned_bboxes.append(chosen_box)

    return cleaned_bboxes






