import cv2
import numpy as np
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(r"C:\Users\Admin\OneDrive\Desktop\3 курс 2 семестр\Компьютерная графика и распознавание образов ТК\coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
image = cv2.imread("lab6.jpg")
max_size = 800
height, width, _ = image.shape
scale = max_size / max(height, width)
new_width = int(width * scale)
new_height = int(height * scale)
image = cv2.resize(image, (new_width, new_height))

blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  
            center_x, center_y, w, h = (detection[:4] * [new_width, new_height, new_width, new_height]).astype("int")
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in indices.flatten():
    x, y, w, h = boxes[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("YOLO Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
