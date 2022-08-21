import torch
from glob import glob
import os
from PIL import Image, ImageOps
import cv2
import random
from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils import iou, augment_image, generate_images


class PascalDataset(Dataset):
    def __init__(self, img_shape, anchors, num_anchors, num_classes=20, S=13, train=True):
        self.anchors = anchors
        self.train = train
        self.img_shape = img_shape
        self.S = S
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.pairs = pd.read_csv('data/train.csv' if train else 'data/test.csv')

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, index):
        img_size = self.img_shape[0]
        path, label_path = self.pairs.iloc[index]
        path = os.path.join('data', 'images', path)
        label_path = os.path.join('data', 'labels', label_path)

        img = Image.open(path).resize(self.img_shape)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

        # S x S x 5 x (p_o, x, y, w, h, 0, 0, ... class ..., 0, 0)
        target = torch.zeros(size=(self.S, self.S, self.num_anchors, 6))
        target[:, :, :, 0] = .05
        with open(label_path, 'r') as file:
            for line in file.readlines():
                c, x, y, width, height = [float(n) for n in line.split()]

                i = int(self.S * y)
                j = int(self.S * x)

                x_inner = self.S * x - j
                y_inner = self.S * y - i
                relative_w = self.S * width
                relative_h = self.S * height

                box = torch.tensor([[0, 0, width, height]])
                ious = iou(self.anchors, box)
                anchor_indices = torch.argsort(ious, descending=True)
                relevant_anchor = anchor_indices[0]
                target[i, j, relevant_anchor, :] = torch.tensor([.95, x_inner, y_inner, relative_w, relative_h, c])

        return transform(img), target






