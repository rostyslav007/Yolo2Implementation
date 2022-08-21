# Yolo2Implementation

Implementation of Yolo2 Darknet architecture and main training components for object detection task based on YOLOv1, YOLOv2 papers

## Project structure:
  - Anchor Boxes.ipynb: notebook for prior anchor boxes search using **K-Means clastering** algorithm
  - loss.py: **YOLO loss function** implementation
  - model.py: **Darknet architecture** model
  - mAP.py: **Mean Average Precision** metric
  - dataset: **Dataset loader** for YOLO training
  - utils: additional functions for **iou(intersection over union)** calculation, data and bboxes **augmentation** and **non max suppression**  
  
## PascalVOCO dataset
http://host.robots.ox.ac.uk/pascal/VOC/

## Papers:
  - You only look once (YOLO): https://arxiv.org/pdf/1506.02640.pdf
  - YOLO9000, better, faster, stronger: https://arxiv.org/pdf/1612.08242v1.pdf
