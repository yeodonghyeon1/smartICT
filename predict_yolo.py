from ultralytics import YOLO
import os

model = YOLO("./model/yolo.pt")  # pretrained YOLOv8n model


def predict_image(img):
    predict_data = ""
    # Run batched inference on a list of images
    results = model(img)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        # print(boxes.xywh.item())
    for i in range(0, len(boxes.cls)):
        pred = "pred " + str(boxes.cls[i].item()) + " "
        box_list = []
        for _ in boxes.xywh:
            for j in range(0, len(_)):
                box_list.append(_[j].item())
        box = "box " + str(box_list[2]) + " " + str(box_list[3]) + " "
        predict_data += pred
        predict_data += box
    print(predict_data)
    return predict_data


# a = os.listdir(r"C:\Users\user\Desktop\dataset\yolov8\train\images")
# for i in a:
#     # print(i)
#     predict_image(r"C:\Users\user\Desktop\dataset\yolov8\train\images" + "/" + i)