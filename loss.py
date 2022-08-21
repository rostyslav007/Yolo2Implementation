import torch
import torch.nn as nn


class YoloLoss(nn.Module):

    def __init__(self, anchors, S):
        super(YoloLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.crossentropy_loss = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.anchors = anchors[:, 2:] * S
        self.num_anchors = len(anchors)
        self.l_coord = 5
        self.l_noobj = 5

    def forward(self, prediction, truth):
        p_b, p_v, p_s, p_s = prediction.shape
        # B S S An 6
        prediction = torch.reshape(prediction, [p_b, self.num_anchors, -1, p_s, p_s]).permute(0, 3, 4, 1, 2)
        obj = truth[..., 0] == .95
        noobj = truth[..., 0] == .05

        # Center coordinate loss
        center_loss = self.mse_loss(self.sigmoid(prediction[obj][:, 1:3]), truth[obj][:, 1:3])

        # Box size loss
        prediction[..., 3:5] = torch.exp(prediction[..., 3:5]) * self.anchors.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        prediction_box = prediction[obj][:, 3:5]
        box_loss = self.mse_loss(prediction_box**0.5, truth[obj][:, 3:5]**0.5)

        # Confidence of object loss
        confidence_obj_loss = self.bce_loss(prediction[obj][:, 0], truth[obj][:, 0])

        # Confidence of noobject loss
        confidence_noobj_loss = self.bce_loss(prediction[noobj][:, 0], truth[noobj][:, 0])

        # classification loss
        classification_loss = self.crossentropy_loss(prediction[obj][:, 5:], truth[obj][:, -1].to(dtype=torch.long))

        loss = (
            self.l_coord * center_loss +
            self.l_coord * box_loss +
            confidence_obj_loss +
            self.l_noobj * confidence_noobj_loss +
            classification_loss
        )

        print(center_loss.item(), box_loss.item(), confidence_obj_loss.item(),
              confidence_noobj_loss.item(), classification_loss.item())

        return loss
