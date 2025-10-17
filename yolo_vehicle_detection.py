import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolo-coco/yolov3-tiny.weights", "yolo-coco/yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load Image
img = cv2.imread('car.jpg')
if img is None:
    print("❌ Image not found! Make sure 'car.jpg' exists in the folder.")
    exit()

height, width = img.shape[:2]

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

# Analyze detections
boxes, confidences, class_ids = [], [], []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and classes[class_id] in ["car", "truck", "bus", "motorbike"]:
            center_x, center_y = int(detection[0] * width), int(detection[1] * height)
            w, h = int(detection[2] * width), int(detection[3] * height)
            x, y = int(center_x - w / 2), int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Remove overlaps
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw the boxes
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# Resize image to larger dimensions (e.g., 2.5x larger)
img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)

# Save and show result
cv2.imwrite("output/result.jpg", img)
cv2.imshow("YOLO Vehicle Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# python yolo_vehicle_detection.py  