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

category_ids = [70, 69, 68, 43]
json_path='../dataset/train.json'
image_dir='../dataset/images'

coco = COCO(json_path)
ann_ids = coco.getAnnIds(catIds=category_ids)
anns = coco.loadAnns(ann_ids)
img_ids = [ann['image_id'] for ann in anns]
img_ids = list(set(img_ids))
print(anns)
for i in range(0, len(img_ids)):
    img = coco.loadImgs(img_ids)[i].copy()
    img_open = Image.open(f'{image_dir}/{img["file_name"]}')
    name = img["file_name"]
    img = img_open
    
    width, height = img.size
    
    # Whichever is bigger, let's make it 320
    if width > height:
        img = img.resize((320, int(320 * height / width)))
    else:
        img = img.resize((int(320 * width / height), 320))
    
    # Make image a tensor and pad it (centered)
    img = torchvision.transforms.ToTensor()(img)
        
    anns[i]['bbox'][0] = anns[i]['bbox'][0] * img.size(2) / width
    anns[i]['bbox'][1] = anns[i]['bbox'][1] * img.size(1) / height
    anns[i]['bbox'][2] = anns[i]['bbox'][2] * img.size(2) / width
    anns[i]['bbox'][3] = anns[i]['bbox'][3] * img.size(1) / height
    print(anns[i]['bbox'][2] * anns[i]['bbox'][3])
    if anns[i]['bbox'][2] * anns[i]['bbox'][3] < 500:
        img_open.save(f"../dataset/size_adjust_images/{name}")

    
    # img_open.save(f"../dataset/adjust_images/{name}")