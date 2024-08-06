import torch
import torch.quantization
import torchvision
from functions import COCODataset, categories, plot_output_image_with_annotations

# run on gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load model from file
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(num_classes=len(categories) + 1, trainable_backbone_layers=6)
model.load_state_dict(torch.load('./models/model_epoch_8.pth'))
model.to(device)
model.eval()

# test it works
def collate_fn(batch):
    return tuple(zip(*batch))

testing_dataset = COCODataset(image_dir='./images', json_path='./test.json', category_ids=categories, device=device)
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

for images, targets in testing_dataloader:
    prediction = model(images)
    
    # plot the image
    plot_output_image_with_annotations(images[0], [prediction[0]['boxes'].cpu().detach().numpy()[0]])
    
    # get the weighted most likely class
    class_weights = [0, 0, 0, 0, 0]
    for i in range(len(prediction[0]['scores'])):
        class_weights[prediction[0]['labels'][i]] = class_weights[prediction[0]['labels'][i]] + prediction[0]['scores'][i]
    
    # get the most likely class (highest of list)
    print("Predicted class: ", class_weights.index(max(class_weights)))
    
    # print actual classes
    print("Actual classes: ", targets[0]['labels'])
    print(class_weights)