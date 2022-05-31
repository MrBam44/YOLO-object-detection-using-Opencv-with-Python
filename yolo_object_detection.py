import cv2 as cv
import numpy as np
from wandb import Classes

# load yolo
net = cv.dnn.readNet("D:\OPEN CV\YOLOV object detection\yolov3.weights",
                     "D:\OPEN CV\YOLOV object detection\yolov3.cfg")
clasees = []
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# print(classes)
layer_name = net.getLayerNames()
output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load Image
img = cv.imread("D:/OPEN CV/YOLOV object detection/room_ser.jpg")
img = cv.resize(img, None, fx=0.4, fy=0.4)
height, width, channel = img.shape

# Detect Objects
blob = cv.dnn.blobFromImage(
    img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layer)
# print(outs)

# Showing Information on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detection
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # cv.circle(img, (center_x, center_y), 10, (0, 255, 0), 2 )
            # Reactangle Cordinate
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# print(len(boxes))
# number_object_detection = len(boxes)

indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)

font = cv.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        # print(label)
        color = colors[i]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(img, label, (x, y + 30), font, 3, color, 3)

cv.imshow("IMG", img)
cv.waitKey(0)
cv.destroyAllWindows()
