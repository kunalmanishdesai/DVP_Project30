import json
from roboflow import Roboflow
import cv2

# Initialize Roboflow
rf = Roboflow(api_key="MVvAEpB4uhnRulRMx1rd")

# Get the project, model, and perform predictions
project = rf.workspace().project("human-detection-iy82e")
model = project.version(1).model


image_path = "fire_uncle_clear.png"
predictions= model.predict(image_path, confidence=40, overlap=30).json()

print(predictions)
# Parse predictions string to a dictionary

# Load the image using OpenCV
image = cv2.imread(image_path)

# Draw rectangles around detected faces
for prediction in predictions['predictions']:
    x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
    cv2.rectangle(image, (x - width//2, y-height//2), (x + width//2, y + height//2), (0, 255, 0), 2)

# Save the image with rectangles drawn around faces
cv2.imwrite("output_with_rectangles.png", image)