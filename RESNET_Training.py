import json
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # Import tqdm for progress bars

# Select the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the ResNet model and move it to the device
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
resnet.eval()

# Function to extract features from frames of a video clip
def extract_features(video_clip):
    frames = []
    cap = cv2.VideoCapture(video_clip)

    if not cap.isOpened():
        print(f"Failed to open video: {video_clip}")
        return np.array([])  # Return empty array if video can't be opened

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print(f"No frames extracted from video: {video_clip}")
        return np.array([])

    frames = np.array(frames)
    feature_vectors = []

    with torch.no_grad():
        for frame in frames:
            input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Move to GPU
            feature = resnet(input_tensor)
            feature_vectors.append(feature.cpu().numpy())  # Move result back to CPU

    return np.array(feature_vectors)

# Load Label annotations from folder names
def get_label_from_folder(folder_name):
    return folder_name

# Specify the root directory containing the extracted folders
root_directory = "/home/kothari.je/videos/70_Videos_Each"

# Initialize lists to store features and labels
all_features = []
all_labels = []

# Iterate through each folder inside the extracted directory
for folder_name in tqdm(os.listdir(root_directory), desc="Processing folders"):
    folder_path = os.path.join(root_directory, folder_name)
    if os.path.isdir(folder_path):  # Ensure it is a folder
        # Use folder name as the label
        label = get_label_from_folder(folder_name)

        # Iterate through video files in the folder
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

        if not video_files:
            print(f"No video files found in folder: {folder_name}")
            continue

        for video_file in tqdm(video_files, desc=f"Processing videos in {folder_name}", leave=False):
            video_clip_path = os.path.join(folder_path, video_file)
            features = extract_features(video_clip_path)

            if features.size == 0:
                print(f"No features extracted from video: {video_file}")
                continue

            all_features.append(features)
            all_labels.extend([label] * features.shape[0])  # Add the label for each frame

# Check if any features were extracted
if not all_features:
    print("No features extracted from any video files.")
    exit()

# Convert features to tensor and move to the device
print(f"Number of feature arrays: {len(all_features)}")
all_features = torch.tensor(np.concatenate(all_features), dtype=torch.float32).to(device)

# Create a mapping from labels to integers
unique_labels = list(set(all_labels))
print("Unique labels:", unique_labels)

label_to_index = {label: index for index, label in enumerate(unique_labels)}
print("Mapping from labels to integers:", label_to_index)

# Convert all_labels to integers
numeric_labels = [label_to_index[label] for label in all_labels]
# print(numeric_labels)
all_labels = torch.tensor(numeric_labels, dtype=torch.long).to(device)  # Convert to tensor and move to GPU

# Calculate the number of unique classes for the output layer
output_size = len(unique_labels)

print("Feature extraction completed.")
print(all_features.cpu().shape)

# Save features and labels to disk
save_path = "videos/features/labels_70_Each.pt"  # Specify the save path
torch.save({
    'features': all_features.cpu(),  # Move back to CPU before saving
    'labels': all_labels.cpu(),      # Move back to CPU before saving
    'label_to_index': label_to_index,
    'unique_labels': unique_labels
}, save_path)

print(f"Features and labels saved to {save_path}.")
