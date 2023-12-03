import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import cv2
import json
from roboflow import Roboflow

snapshots_path = "snapshots"

# Initialize Roboflow
rf = Roboflow(api_key="II6omqQUqNbP86ZLcAa9")

# Get the project, model
project = rf.workspace().project("human-detection-iy82e")
model = project.version(1).model


def real_time_dehaze_and_save_video(video_path, output_path):
    # Load the dehaze network and move it to the appropriate device
    dehaze_net = net.dehaze_net().to('mps' if torch.backends.mps.is_available() else 'cpu')
    dehaze_net.load_state_dict(torch.load(snapshots_path + '/dehazer.pth', map_location='mps' if torch.backends.mps.is_available() else 'cpu'))
    dehaze_net.eval()  # Set the network to evaluation mode

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2  # Double the width for side-by-side video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # or other codec compatible with your video

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to a PyTorch tensor
        frame_tensor = torch.from_numpy(frame / 255.0).float()
        frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor.unsqueeze(0).to('mps' if torch.backends.mps.is_available() else 'cpu')

        # Dehaze the frame
        with torch.no_grad():  # No need to track gradients
            clean_frame_tensor = dehaze_net(frame_tensor)

        # Convert back to numpy array
        clean_frame = clean_frame_tensor.squeeze().detach().permute(1, 2, 0).cpu().numpy()
        clean_frame = (clean_frame * 255).astype(np.uint8)

        # Convert numpy array to cv2.Mat format
        clean_frame_cv2 = cv2.cvtColor(clean_frame, cv2.COLOR_RGB2BGR)

        # Perform predictions on the frame
        predictions = model.predict(clean_frame_cv2, confidence=40, overlap=30).json()

        # Draw rectangles around detected humans
        for prediction in predictions['predictions']:
            x, y, width, height = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
            
            # Calculate bounding box coordinates
            x1 = int(x - width // 2)
            y1 = int(y - height // 2)
            x2 = int(x + width // 2)
            y2 = int(y + height // 2)

            cv2.rectangle(clean_frame_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert back to numpy array for displaying or writing to video
        clean_frame_with_rectangles = cv2.cvtColor(clean_frame_cv2, cv2.COLOR_BGR2RGB)

        # Concatenate original and dehazed frames
        combined_frame = cv2.hconcat([frame, clean_frame_with_rectangles])

        # Write the combined frame to the output video
        out.write(combined_frame)

        # # Display the combined frame
        # cv2.imshow('Original and Dehazed Video', combined_frame)

        # # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    real_time_dehaze_and_save_video('test_video/fire_girl_short.mp4', 'results/test7.mp4')