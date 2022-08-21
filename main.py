import torch
from utils import iou, load_anchors, non_max_suppression
from model import TiniYolo2
from loss import YoloLoss
from dataset import PascalDataset
import torch.optim as optim
from config import *
from torch.utils.data import DataLoader
from tqdm import tqdm

anchors = load_anchors(anchors_path)
num_anchors = len(anchors)

train_dataset = PascalDataset(img_shape=img_shape, anchors=anchors, num_anchors=num_anchors, train=True)
test_dataset = PascalDataset(img_shape=img_shape, anchors=anchors, num_anchors=num_anchors, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = TiniYolo2(num_anchors=num_anchors, in_channels=3, num_classes=num_classes).to(device)
if load_pretrained:
    model.load_state_dict(torch.load(load_path))

criterion = YoloLoss(anchors=anchors.to(device), S=S)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
scaler = torch.cuda.amp.GradScaler()

if training:
    for e in tqdm(range(num_epochs)):
        print('iteration: ', e)
        losses_history = []
        for images, targets in train_loader:
            with torch.cuda.amp.autocast():
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                loss = criterion(preds, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses_history.append(loss.item())

            print('loss: ', losses_history[-1])

        torch.save(model.state_dict(), MODEL_PATH)
        # Calculating test loss
        with torch.no_grad():
            err = 0
            count = 0
            for imgs, targets in test_loader:
                with torch.cuda.amp.autocast():
                    imgs, targets = imgs.to(device), targets.to(device)
                    preds = model(imgs)
                    loss = criterion(preds, targets)

                err += loss.item()
                count += 1
                if count > 20:
                    break
            print('Test loss: ', err / count, ' Train loss: ', sum(losses_history) / len(losses_history))

else:
    # calculation mAP metric
    model.load_state_dict(torch.load(load_path))
    model = model.to(device)
    losses = []
    for imgs, targets in test_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss = criterion(preds, targets)

        losses.append(loss.item())

    print('Test loss: ', sum(losses) / len(losses))
