import torch
import os
from glob import glob

num_classes = 20
training = False
load_pretrained = True
MODEL_PATH = 'model/PascalYolo2_v3.pt'
anchors_path = 'data/anchors_5.txt'
load_path = 'model/PascalYolo2_v2_1.pt'
img_shape = (416, 416)
device = torch.device('cuda' if (torch.cuda.is_available() and training) else 'cpu')
lr = 1e-4
lr_decrease_rate = 0.5
S = 15
batch_size = 16
num_epochs = 160
