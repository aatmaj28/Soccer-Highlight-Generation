import os
import cv2
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
# import torchvision.models as models

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the ResNet model for feature extraction
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
resnet.eval()

# Function to extract features from a video clip
def extract_features(video_clip):
    frames = []
    cap = cv2.VideoCapture(video_clip)

    if not cap.isOpened():
        print(f"Failed to open video: {video_clip}")
        return np.array([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0  # Normalize
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print(f"No frames extracted from video: {video_clip}")
        return np.array([])

    frames = np.array(frames)
    feature_vectors = []

    with torch.no_grad():
        for frame in frames:
            input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
            feature = resnet(input_tensor)
            feature_vectors.append(feature.cpu().numpy())

    return np.array(feature_vectors)

# Load the trained GRU model
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(input_dim, 1, bias=False)
    def forward(self, x):
        attn_weights = self.attn(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted_input = x * attn_weights
        return weighted_input, attn_weights
 
# Enhanced GRU Model (same as in the training script)
class EnhancedGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(EnhancedGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.attn = Attention(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # Ensure input is 3D (batch, sequence, features)
        if x.dim() == 4:
            # If 4D, assume it's (batch, sequence, height, width) from feature extraction
            x = x.view(x.size(0), x.size(1), -1)
        out, _ = self.gru(x)
        weighted_out, attn_weights = self.attn(out)
        out = weighted_out.sum(dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc3(out)

# Load the model checkpoint
checkpoint_path = '/home/kothari.je/videos/model/final_enhanced_gru_model.pth'
checkpoint = torch.load(checkpoint_path)
input_size = checkpoint['input_size']
output_size = checkpoint['output_size']
label_to_index = checkpoint['label_to_index']
index_to_label = {index: label for label, index in label_to_index.items()}

# Instantiate the model
model = EnhancedGRUModel(input_size, hidden_size=2048, output_size=output_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Specify the test folder
test_folder = "/home/kothari.je/test_env/input_videos_and_labels/extracted"

# Create a DataFrame for storing results
results = []

# Iterate through each folder and video for inference
for folder_name in tqdm(os.listdir(test_folder), desc="Processing folders"):
    folder_path = os.path.join(test_folder, folder_name)
    if os.path.isdir(folder_path):  # Check if it is a folder
        true_label = folder_name  # Use folder name as true label
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

        if not video_files:
            print(f"No video files found in folder: {folder_name}")
            continue

        for video_file in tqdm(video_files, desc=f"Processing videos in {folder_name}", leave=False):
            video_path = os.path.join(folder_path, video_file)
            features = extract_features(video_path)

            if features.size == 0:
                print(f"No features extracted from video: {video_file}")
                continue

            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

            with torch.no_grad():
                outputs = model(features_tensor)
                avg_output = torch.mean(outputs, dim=0)  # Average predictions across frames
                predicted_index = torch.argmax(avg_output).item()
                predicted_label = index_to_label[predicted_index]

            results.append({
                "Video File": video_file,
                "True Label": true_label,
                "Predicted Label": predicted_label
            })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("inference_results.csv", index=False)
print("Inference completed. Results saved to inference_results.csv.")
