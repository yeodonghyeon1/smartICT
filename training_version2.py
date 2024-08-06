from pycocotools.coco import COCO
from torchvision import torch
import torch.nn as nn
import torchvision.ops
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
#필요 라이브러리 및 모듈 불러오기

# 디바이스 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from functions import *
# from test import *
def collate_fn(batch):
    return tuple(zip(*batch))

# 데이터셋 설정
testing_dataset = COCODataset(image_dir='../dataset/size_adjust_images', json_path='../dataset/train.json', category_ids=categories, device=device)
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)



# 모델 설정
# model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
#     weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
# )
# model.eval()

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
    weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    num_classes=len(categories) + 1,
    trainable_backbone_layers=6
)

# 학습 설정
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

# 학습률 스케줄러 설정
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 1000
model.to(device)
batch_size = 12

testing_dataloader = torch.utils.data.DataLoader(
    testing_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
)

for epoch in range(num_epochs):
    model.train()
    training_dataset = COCODataset(image_dir='../dataset/size_adjust_images', json_path='../dataset/train.json', category_ids=categories, device=device)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    i = 0
    total_loss = 0
    for images, targets in training_dataloader:
        i += 1
        # 이미지와 타겟을 디바이스로 이동
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += (losses / batch_size)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {total_loss / i}")

    # 학습률 업데이트
    lr_scheduler.step()

    # 테스트 데이터셋에서 평가
    metric = testing_dataset.coco_evaluate(model, testing_dataloader, device)
    print(f"Epoch: {epoch}, mAP: {metric}")

    # 모델 저장
    torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

# 모델 평가 및 시각화
model.eval()


for images, targets in testing_dataloader:

    prediction = model(images)
    print("HERE")

    # 박스 추출 및 렌더링
    boxes = prediction[0]['boxes']
    boxes = boxes.cpu().detach().numpy()

    # 최고 점수의 인덱스 가져오기
    idx = boxes.argmax()
    # plot_output_image_with_annotations(images[0], boxes[idx])

    # 가장 가능성이 높은 클래스 가져오기
    class_weights = [0, 0, 0, 0, 0]
    for i in range(len(prediction[0]['scores'])):
        class_weights[prediction[0]['labels'][i]] += prediction[0]['scores'][i]
    print(class_weights)
    print("HERE")