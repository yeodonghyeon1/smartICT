import torch
import torch.quantization
import torchvision
from functions import COCODataset, categories, plot_output_image_with_annotations
from PIL import Image
import matplotlib.pyplot as plt
# run on gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(num_classes=len(categories) + 1, trainable_backbone_layers=6)
model.load_state_dict(torch.load('./model/model_epoch_2.pth'))
model.to(device)
model.eval()
# load model from file


# test it works
def collate_fn(batch):
    return tuple(zip(*batch))


def predict_image(img):
    # img = Image.open(img)
    img = Image.fromarray(img) 
    # Let's get the dimensions of the image, it will be needed to scale the annotations
    width, height = img.size
    print("aaaaaaaa")
    
    # Whichever is bigger, let's make it 320
    if width > height:
        img = img.resize((320, int(320 * height / width)))
    else:
        img = img.resize((int(320 * width / height), 320))
    
    # Make image a tensor and pad it (centered)
    img = torchvision.transforms.ToTensor()(img)
    
    # Make annotations into expected format
    # First, let's scale the annotations [x, y, width, height]
    # for ann in anns:
    #     ann['bbox'][0] = ann['bbox'][0] * img.size(2) / width
    #     ann['bbox'][1] = ann['bbox'][1] * img.size(1) / height
    #     ann['bbox'][2] = ann['bbox'][2] * img.size(2) / width
    #     ann['bbox'][3] = ann['bbox'][3] * img.size(1) / height
    
    # # Now, convert from [x, y, width, height] to [x1, y1, x2, y2]
    # for ann in anns:
    #     ann['bbox'][2] += ann['bbox'][0]
    #     ann['bbox'][3] += ann['bbox'][1]
        
        # Normalize the image
        # img = torchvision.transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Grayscale the image
        # img = torchvision.transforms.Grayscale()(img)
        # img.repeat(3, 1, 1)
        
    # Pad
    
    img = torch.nn.functional.pad(img, (0, 320 - img.shape[2], 0, 320 - img.shape[1]), value=1)
    img = img.to(device)
    img = torch.unsqueeze(img, 0)
    prediction = model(img)
    print(prediction)
    # plot the image
    
    # get the weighted most likely class
    class_weights = [0, 0, 0, 0, 0]
    for i in range(len(prediction[0]['scores'])):
        class_weights[prediction[0]['labels'][i]] = class_weights[prediction[0]['labels'][i]] + prediction[0]['scores'][i]
    
    # get the most likely class (highest of list)
    print("Predicted class: ",class_weights.index(max(class_weights)))
    print("Predicted class: ",prediction[0]['labels'][0].item())
    return prediction

