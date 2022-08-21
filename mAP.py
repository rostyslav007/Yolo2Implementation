import torch
from collections import Counter
from utils import iou


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    # pred_boxes (list): [[train_idx, class_pred, prob_score, x, y, w, c], ...]
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truth = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[0] == c:
                ground_truth.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truth])

        for key, value in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(value)

        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        total_true_bboxes = len(ground_truth)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truth if bbox[0] == detection[0]
            ]

            num_ground_truth = len(ground_truth_img)
            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(ground_truth_img):
                iou_score = iou(torch.tensor(detection[3:]), torch.tensor(gt[3:]))
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)

            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = torch.divide(TP_cumsum, TP_cumsum + FP_cumsum + epsilon)
            precisions = torch.cat([torch.tensor([1]), precisions])
            recalls = torch.cat([torch.tensor([0]), recalls])
            average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)