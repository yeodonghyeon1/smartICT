from pycocotools.coco import COCO
import torch
import torchvision
import torch.nn as nn
from torchvision.ops import *
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from functions import *

# Make sure we run on the correct device
device = torch.device('cuda')

# dataloaders
def collate_fn(batch):
    return tuple(zip(*batch))

testing_dataset = COCODataset(image_dir='../dataset/images', json_path='../dataset/train.json', category_ids=categories, device=device)
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)



model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
    weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2, 
    num_classes=len(categories) + 1, 
    trainable_backbone_layers=6
)

### Training ###

# Define the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

# Define the learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 10
# move to device
model.to(device)
batch_size = 16

for epoch in range(num_epochs):
    model.train()
    training_dataset = COCODataset(image_dir='../dataset/images', json_path='../dataset/train.json', category_ids=categories, device=device)
    print(training_dataset.coco.imgs)
    print(training_dataset.coco.anns)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    i = 0
    total_loss = 0
    for images, targets in training_dataloader:
        # if bool(targets[0]) == False:
        #     continue
        i += 1

        # move the images and targets to the device
        # prediction = model(images) 
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss = total_loss + (losses / batch_size)
        # print(images)
        # print(targets)
        # print("loss dict", loss_dict , "\n\n\n\n\n\n\n")
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
        # print average loss
        print(f"Epoch: {epoch}, Loss: {total_loss / i}")
    
    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset
    metric = testing_dataset.coco_evaluate(model, testing_dataloader, device)
    print(f"Epoch: {epoch}, mAP: {metric}")
    
    # save the model with epoch number
    torch.save(model.state_dict(), f"./model/model_epoch_{epoch}.pth")

# let's evaluate the model on a test and visualize
# model.eval()

# for images, targets in testing_dataloader:
#     prediction = model(images)

#     # take the boxes and render them
#     boxes = prediction[0]['boxes']
#     boxes = boxes.cpu().detach().numpy()
    
#     # get the index of the max score
#     boxes = [boxes[prediction[0]['scores'].argmax()]]
#     plot_output_image_with_annotations(images[0], boxes)
    
#     # get the weighted most likely class
#     class_weights = [0, 0, 0, 0, 0]
#     for i in range(len(prediction[0]['scores'])):
#         class_weights[prediction[0]['labels'][i]] = class_weights[prediction[0]['labels'][i]] + prediction[0]['scores'][i]
#     print(class_weights)
    