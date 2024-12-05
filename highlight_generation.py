import os
import cv2
import numpy as np
import torch
from collections import Counter, deque
import torch.nn as nn
import torchvision.models as models
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(input_dim, 1, bias=False)  # Linear layer to compute attention scores
    def forward(self, x):
        # Calculate attention scores
        attn_weights = self.attn(x)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Softmax to normalize
        weighted_input = x * attn_weights  # Apply attention weights to the input
        return weighted_input, attn_weights
 
# Enhanced GRU Model with Attention
class EnhancedGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(EnhancedGRUModel, self).__init__()
        # GRU layer with multiple layers and dropout
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        # Attention mechanism
        self.attn = Attention(hidden_size)
        # Fully connected layers with increased hidden size
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)  # Increase the size of the first FC layer
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)  # Keep the second layer the same size
        self.fc3 = nn.Linear(hidden_size, output_size)  # Final output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Dropout layer
    def forward(self, x):
        # GRU output
        out, _ = self.gru(x)
        # Apply Attention
        weighted_out, attn_weights = self.attn(out)
        # Take the output from the last time step (sum over the weighted output)
        out = weighted_out.sum(dim=1)
        # Fully connected layers with ReLU activations and Dropout
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        # Output layer
        return self.fc3(out)
 
# Load the trained model
model_path = '/home/kothari.je/videos/model/final_enhanced_gru_model.pth'
model_data = torch.load(model_path)
 
 
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to('cuda')
resnet.eval()
 
# Extract model parameters
label_to_index = model_data['label_to_index']
index_to_label = {v: k for k, v in label_to_index.items()}
model = EnhancedGRUModel(
    input_size=model_data['input_size'],
    hidden_size=2048,  # This should match the hidden size during training
    output_size=model_data['output_size']
).to(device)
model.load_state_dict(model_data['model_state_dict'])
model.eval()
 
# Function to extract features from a video (as used during training)
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
            input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
            feature = resnet(input_tensor)
            feature_vectors.append(feature.cpu().numpy())
 
    return np.array(feature_vectors)
 
# Function to predict labels for video clips
def predict_label(video_clip):
    features = extract_features(video_clip)
    if features.size == 0:
        return None
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(features_tensor)
        predictions = torch.argmax(output, dim=1)
 
    frame_labels = [index_to_label[p.item()] for p in predictions]
    # print("In predict label function ",predictions, frame_labels)
    # label_counts = Counter(frame_labels)  # Count occurrences of each label
    # most_common_label, _ = label_counts.most_common(1)[0]  # Most frequent label

    # print(most_common_label)
    return frame_labels
 
# Highlight reel creation logic
def create_highlight_reel(video_clips, output_path, clip_length=10, highlight_duration=500):
    total_clips = highlight_duration // clip_length
    data_structure = deque(maxlen=total_clips)
    all_predictions = []
    key_events = {
        "Kick-off": 3,
        "Goal": 1,
        "Shots on target": 2,
        "Red card": 4,
        "Corner": 5,
        "Yellow card": 6,
        "Foul": 7
    }
 
    # print(data_structure)
    # Predict labels for each clip
    for video_clip in video_clips:
        predicted_label = predict_label(video_clip)
        if predicted_label and predicted_label[0] in key_events.keys():
            all_predictions.append((video_clip, predicted_label[0]))  # Store clip and its label
 
    # print(all_predictions)
    # Sort clips by time sequence (assuming clips are named sequentially by time)
    all_predictions.sort(key=lambda x: x[0])

    print("All Clip Predictions : ")
    for clip, label in all_predictions:
        print(f"{clip:<30} | {label}")
 
    # Count goals
    goal_clips = [clip for clip, label in all_predictions if label == "Goal"]
    
    print("Total Goal Predicted Clips : ", goal_clips)
    print("Length of all predictions before ",len(all_predictions))
    
    # # Push clips into the data structure
    # for clip, label in all_predictions[:total_clips]:
    #     if label == "Goal":
    #         data_structure.append((clip, label))

    # all_predictions = all_predictions[total_clips:]
   
    # print("Length of all predictions after ",len(all_predictions))

    for g in goal_clips: 
        data_structure.append((g,"Goal"))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = None
 
    for clip, _ in data_structure:
        cap = cv2.VideoCapture(clip)
        if not cap.isOpened():
            print(f"Failed to open clip: {clip}")
            continue
 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
 
            # Initialize VideoWriter with the correct frame size
            if output_video is None:
                height, width, _ = frame.shape
                output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
 
            output_video.write(frame)
 
        cap.release()
 
    if output_video is not None:
        output_video.release()
        print(f"Highlight reel created: {output_path}")
    else:
        print("Failed to create highlight reel. No valid frames were written.")
 
# Example Usage
video_clips = [f"/home/kothari.je/videos/output/{file}" for file in os.listdir("/home/kothari.je/videos/output") if file.endswith(('.mp4', '.avi'))]
output_path = "highlight_reel.mp4"
create_highlight_reel(video_clips, output_path)