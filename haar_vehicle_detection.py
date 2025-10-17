import cv2
import os

# Load Haar cascade
car_cascade = cv2.CascadeClassifier('cascade/cars.xml')

# Load image
img_path = os.path.join(os.getcwd(), "car.jpg")
img = cv2.imread(img_path)

if img is None:
    print("Error: Image not found or could not be read.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect cars
cars = car_cascade.detectMultiScale(gray, 1.1, 3)

# Draw rectangles
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show image
cv2.imshow('Vehicle Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#python haar_vehicle_detection.py