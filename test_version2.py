import torch
import torch.quantization
import torchvision
from functions import COCODataset, categories, plot_output_image_with_annotations
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
# GPU 사용 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 파일에서 모델 로드
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
    num_classes=len(categories) + 1,
    trainable_backbone_layers=6
)
model.load_state_dict(torch.load('./model/model_epoch_101.pth'))
model.to(device)
model.eval()

# 작동 테스트
def collate_fn(batch):
    return tuple(zip(*batch))

testing_dataset = COCODataset(image_dir='../dataset/images', json_path='../dataset/test.json', category_ids=categories, device=device)
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

all_preds = []
all_targets = []

for images, targets in testing_dataloader:
    images = [image.to(device) for image in images]
    prediction = model(images)

    # 이미지 출력
    # plot_output_image_with_annotations(
    #     images[0].cpu(),
    #     [prediction[0]['boxes'].cpu().detach().numpy()[0]]
    # )

    # 가장 가능성이 높은 클래스 계산
    class_weights = [0] * (len(categories) + 1)  # Add one more element to prevent index out of range
    for i in range(len(prediction[0]['scores'])):
        label = prediction[0]['labels'][i].item()
        if label < len(class_weights):  # Check to prevent index out of range
            class_weights[label] += prediction[0]['scores'][i].item()
    predicted_class = class_weights.index(max(class_weights))
    
    # predicted_class = prediction[0]['labels'][0].item()
    actual_class = targets[0]['labels'].cpu().numpy()

    # 예측 라벨과 실제 라벨 저장
    all_preds.append(predicted_class)
    all_targets.append(actual_class[0])

    # 가장 가능성이 높은 클래스 출력
    print("Predicted class: ", predicted_class)
    # 실제 클래스 출력
    print("Actual class: ", actual_class[0])
    print(class_weights)
# Confusion Matrix 계산 및 시각화
cf_matrix = confusion_matrix(all_targets, all_preds)

print("Precision Score : ",precision_score(all_targets, all_preds, 
                                           pos_label='positive',
                                           average='micro'))
print("Recall Score : ",recall_score(all_targets, all_preds, 
                                           pos_label='positive',
                                           average='micro'))
# f1_score(all_targets, all_preds)
print("cf_matrix", cf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
