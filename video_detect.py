import json
from roboflow import Roboflow
import cv2


# Open video capture
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform predictions on the frame
    predictions = model.predict(frame, confidence=40, overlap=30).json()

    # Draw rectangles around detected humans
    # for prediction in predictions['predictions']:
    #     x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
    #     cv2.rectangle(frame, (x - width//2, y - height//2), (x + width//2, y + height//2), (0, 255, 0), 2)
    for prediction in predictions['predictions']:
        x, y, width, height = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
        
        # Calculate bounding box coordinates
        x1 = int(x - width // 2)
        y1 = int(y - height // 2)
        x2 = int(x + width // 2)
        y2 = int(y + height // 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the annotated frame in real-time
    cv2.imshow('Annotated Frame', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close the window
cap.release()
cv2.destroyAllWindows()