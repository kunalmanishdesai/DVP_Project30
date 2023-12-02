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

snapshots_path = "snapshots"

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

        # Concatenate original and dehazed frames
        combined_frame = cv2.hconcat([frame, clean_frame])

        # Write the combined frame to the output video
        out.write(combined_frame)

        # Display the combined frame
        cv2.imshow('Original and Dehazed Video', combined_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def real_time_dehaze_video(video_path):
    # Load the dehaze network and move it to the appropriate device
    dehaze_net = net.dehaze_net().to('mps' if torch.backends.mps.is_available() else 'cpu')
    dehaze_net.load_state_dict(torch.load(snapshots_path + '/dehazer.pth', map_location='mps' if torch.backends.mps.is_available() else 'cpu'))
    dehaze_net.eval()  # Set the network to evaluation mode

    # Open the video file
    cap = cv2.VideoCapture(video_path)

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
	
        # Display the dehazed frame
        cv2.imshow('Dehazed Video', clean_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def dehaze_video(video_path, output_path):
    # Load the dehaze network and move it to the appropriate device
    dehaze_net = net.dehaze_net().to('mps' if torch.backends.mps.is_available() else 'cpu')
    dehaze_net.load_state_dict(torch.load(snapshots_path + '/dehazer.pth', map_location='mps' if torch.backends.mps.is_available() else 'cpu'))
    dehaze_net.eval()  # Set the network to evaluation mode

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
        clean_frame_tensor = dehaze_net(frame_tensor)
        
		# Convert back to numpy array and write to the output video
        clean_frame = clean_frame_tensor.squeeze().detach().permute(1, 2, 0).cpu().numpy()
        clean_frame = (clean_frame * 255).astype(np.uint8)
        out.write(clean_frame)
    
    cap.release()
    out.release()


def dehaze_image(image_path):

	data_hazy = Image.open(image_path)
	data_hazy = (np.asarray(data_hazy)/255.0)

	data_hazy = torch.from_numpy(data_hazy).float()
	data_hazy = data_hazy.permute(2,0,1)

	data_hazy = data_hazy.unsqueeze(0).to('mps' if torch.backends.mps.is_available() else 'cpu')

	dehaze_net = net.dehaze_net().to('mps' if torch.backends.mps.is_available() else 'cpu')
	dehaze_net.load_state_dict(torch.load(snapshots_path + '/dehazer.pth',map_location='mps' if torch.backends.mps.is_available() else 'cpu'))

	clean_image = dehaze_net(data_hazy)
	torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "results/" + image_path.split("/")[-1])
	

if __name__ == '__main__':

	# test_list = glob.glob("test_images/*")

	# for image in test_list:

	# 	dehaze_image(image)
	# 	print(image, "done!")

	# Example usage
	# dehaze_video('test_video/hazy_video.mp4', 'results/dehazed_video.mp4')
	# real_time_dehaze_video('test_video/hazy_video.mp4')

	real_time_dehaze_and_save_video('test_video/test4.mp4', 'results/test4.mp4')
