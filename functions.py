import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset
import torch
import torchvision
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
def plot_image_with_annotations(image, annotations):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for annotation in annotations:
        bbox = annotation['bbox']  # let's display [x0, y0, x1, y1]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    plt.savefig('input.png')

def plot_output_image_with_annotations(image, annotations):
    # render the image with the annotations
    image = torchvision.transforms.ToPILImage()(image.cpu())
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for annotation in annotations:
        # let's display [x0, y0, x1, y1]
        rect = patches.Rectangle(
            (annotation[0], annotation[1]),
            annotation[2] - annotation[0],
            annotation[3] - annotation[1],
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    plt.savefig('output.png')
    
class COCODataset(Dataset):
    def __init__(self, image_dir, json_path, category_ids, device='cpu'):
        # Perform the initial filtering of the dataset to only the images we want
        coco = COCO(json_path)
        ann_ids = coco.getAnnIds(catIds=category_ids)
        anns = coco.loadAnns(ann_ids)
        img_ids = [ann['image_id'] for ann in anns]
        image_list = os.listdir(image_dir)
        image_list = [int(image.replace(".jpg", "")) -1 for image in image_list]
        img_ids = list(set(img_ids))
        img_ids = [i for i in img_ids if i in image_list]
        self.device = device
        
        # Save the relevant info
        self.json_path = json_path
        self.coco = coco
        self.category_ids = category_ids
        self.image_dir = image_dir
        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.category_ids).copy()
        anns = self.coco.loadAnns(ann_ids).copy()
        img = self.coco.loadImgs(img_id)[0].copy()
        img = Image.open(f'{self.image_dir}/{img["file_name"]}')
        saved_img = img.copy()
        
        # Let's get the dimensions of the image, it will be needed to scale the annotations
        width, height = img.size
        
        # Whichever is bigger, let's make it 320
        if width > height:
            img = img.resize((320, int(320 * height / width)))
        else:
            img = img.resize((int(320 * width / height), 320))
        
        # Make image a tensor and pad it (centered)
        img = torchvision.transforms.ToTensor()(img)
        

        # toSmallImg = False
        # Make annotations into expected format
        # First, let's scale the annotations [x, y, width, height]
        for ann in anns:
            ann['bbox'][0] = ann['bbox'][0] * img.size(2) / width
            ann['bbox'][1] = ann['bbox'][1] * img.size(1) / height
            ann['bbox'][2] = ann['bbox'][2] * img.size(2) / width
            ann['bbox'][3] = ann['bbox'][3] * img.size(1) / height
            # print(ann['bbox'][2] * ann['bbox'][3])
        #     if ann['bbox'][2] * ann['bbox'][3] < 500:
        #         toSmallImg = True
        # if toSmallImg == True:
        #     temp = {}
        #     return 1, temp 
        # Now, convert from [x, y, width, height] to [x1, y1, x2, y2]
        for ann in anns:
            ann['bbox'][2] += ann['bbox'][0]
            ann['bbox'][3] += ann['bbox'][1]
        # Normalize the image
        # img = torchvision.transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Grayscale the image
        # img = torchvision.transforms.Grayscale()(img)
        # img.repeat(3, 1, 1)
        
        # Pad
        img = torch.nn.functional.pad(img, (0, 320 - img.shape[2], 0, 320 - img.shape[1]), value=1)
        
        # Category shift
        for ann in anns:
            ann['category_id'] = self.category_ids.index(ann['category_id']) + 1
        
        # Let's visualize our squished image and annotations to verify
        converted_image = torchvision.transforms.ToPILImage()(img)
        # print(anns)

        # plot_image_with_annotations(converted_image, anns)
        # Finally, convert to the record format expected by the model
        anns = { 
            'boxes': torch.tensor([ann['bbox'] for ann in anns], dtype=torch.float).to(self.device),
            'labels': torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64).to(self.device),
            'image_id': torch.tensor([img_id], dtype=torch.int64).to(self.device),
        }
        
        # Move image to device
        img = img.to(self.device)
        # print(type(img), type(anns))
        return img, anns

    def prepare_for_coco_detection(self, predictions, dataset):
        coco_results = []
        
        for id in enumerate(predictions):
            id = id[1]
            boxes = predictions[id]["boxes"]
            scores = predictions[id]["scores"]
            labels = predictions[id]["labels"]
            boxes = boxes.tolist()
            scores = scores.tolist()
            labels = labels.tolist()
            
            coco_results.extend([
                {
                    "image_id": id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ])
        
        return coco_results

    def coco_evaluate(self, model, data_loader, device):
        model.eval()
        coco = COCO(self.json_path)  # Path to your COCO annotations file
        coco_results = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                # if bool(targets[0]) == False:
                #     continue
                images = list(image.to(device) for image in images)
                outputs = model(images)
                res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                coco_results.extend(self.prepare_for_coco_detection(res, data_loader.dataset))
        
        coco_dt = coco.loadRes(coco_results)
        coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats[0]  # mAP

# Create a COCODataset object
categories = [70, 69, 68, 43]