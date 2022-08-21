import torch
import torch.nn as nn


class TiniYolo2(nn.Module):
    # expected image size 256x256
    def __init__(self, num_anchors, num_classes=1, in_channels=1):
        super(TiniYolo2, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        out_channels = [32*2**p for p in range(6)]
        model_list = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels[1], kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out_channels[1]),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ])
        for i in range(1, len(out_channels) - 2):
            model_list.append(nn.Conv2d(in_channels=out_channels[i], out_channels=out_channels[i+1],
                                        kernel_size=3, padding='same'))
            model_list.append(nn.BatchNorm2d(num_features=out_channels[i+1]))
            model_list.append(nn.LeakyReLU(0.1))
            model_list.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.tail = nn.Sequential(*model_list)

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=out_channels[-2], out_channels=out_channels[-2], kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out_channels[-2]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=out_channels[-2], out_channels=out_channels[-2], kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out_channels[-2]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=out_channels[-2], out_channels=out_channels[-1], kernel_size=2, stride=2)
        )

        self.prediction_head = nn.Sequential(
            nn.Conv2d(in_channels=out_channels[-2]*4 + out_channels[-1],
                      out_channels=self.num_anchors * (5 + self.num_classes),
                      kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, x):
        tail = self.tail(x)
        b, c, h, w = tail.shape
        skip = torch.reshape(tail, (b, c*4, h // 2, w // 2))
        body = self.body(tail)
        concatanated = torch.concat([body, skip], dim=1)
        final = self.prediction_head(concatanated)

        return final


