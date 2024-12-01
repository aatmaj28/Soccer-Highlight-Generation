# import cv2
# import torch
# import heapq
# import numpy as np
# import torch.nn as nn
# import torchvision.models as models
 
# # Load the trained model and ResNet
# model_path = 'videos/model/best_enhanced_gru_model_62_each.pth'
# saved_model_data = torch.load(model_path)
# class EnhancedGRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
#         super(EnhancedGRUModel, self).__init__()
        
#         # Add multiple GRU layers
#         self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
#         # Fully connected layers with increased hidden size
#         self.fc1 = nn.Linear(hidden_size, hidden_size * 2)  # Increase the size of the first FC layer
#         self.fc2 = nn.Linear(hidden_size * 2, hidden_size)  # Keep the second layer the same size
#         self.fc3 = nn.Linear(hidden_size, output_size)  # Final output layer
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)  # Dropout layer
    
#     def forward(self, x):
#         out, _ = self.gru(x)
#         out = out[:, -1, :]  # Take the output from the last time step
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.dropout(out)  # Apply dropout
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.dropout(out)  # Apply dropout again
#         return self.fc3(out)
 
# # Recreate model and load weights
# input_size = saved_model_data['input_size']
# output_size = saved_model_data['output_size']
# hidden_size = 2048
# model = EnhancedGRUModel(input_size, hidden_size, output_size).to('cuda')
# model.load_state_dict(saved_model_data['model_state_dict'])
# model.eval()
 
# label_to_index = saved_model_data['label_to_index']
# unique_labels = saved_model_data['unique_labels']
# index_to_label = {v: k for k, v in label_to_index.items()}
 
# resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to('cuda')
# resnet.eval()

# def extract_features(video_clip):
#     frames = []
#     cap = cv2.VideoCapture(video_clip)
 
#     if not cap.isOpened():
#         print(f"Failed to open video: {video_clip}")
#         return np.array([])
 
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (224, 224))
#         frame = frame / 255.0  # Normalize
#         frames.append(frame)
 
#     cap.release()
 
#     if len(frames) == 0:
#         print(f"No frames extracted from video: {video_clip}")
#         return np.array([])
 
#     frames = np.array(frames)
#     feature_vectors = []
 
#     with torch.no_grad():
#         for frame in frames:
#             input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
#             feature = resnet(input_tensor)  # Extract features
#             feature = feature.squeeze()  # Remove batch dimension (1, 2048)
#             feature_vectors.append(feature.cpu().numpy())
 
#     return np.stack(feature_vectors)  # Stack into a 3D tensor (num_frames, features_dim)


# def predict_labels(video_path, clip_duration=20):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     total_duration = total_frames // fps
#     cap.release()
 
#     predictions = []
#     for start_time in range(0, total_duration, clip_duration):
#         # Extract features from the 20-second clip
#         features = extract_features(video_path)
#         if features.size == 0:
#             continue
 
#         # Convert to tensor and add batch dimension
#         features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
 
#         with torch.no_grad():
#             outputs = model(features_tensor)  # Pass through GRU
#             avg_output = torch.mean(outputs, dim=1)  # Average across sequence
#             predicted_index = torch.argmax(avg_output).item()
#             predicted_label = index_to_label[predicted_index]
 
#         predictions.append((start_time, start_time + clip_duration, predicted_label))
 
#     return predictions 



# # Highlight reel generation using OpenCV
# def generate_highlight_reel(predictions, video_path, target_duration=5 * 60, clip_duration=20):
#     ranking_dict = {
#         "goal": 5,
#         "penalty": 4,
#         "foul": 3,
#         "corner": 2,
#         "other": 1,
#     }
#     clip_length = clip_duration
#     max_clips = target_duration // clip_length
 
#     # Count goals and sort predictions by time
#     goals = [pred for pred in predictions if pred[2] == "goal"]
#     predictions.sort(key=lambda x: x[0])
 
#     # Priority queue for maintaining the data structure
#     selected_clips = []
#     heapq.heapify(selected_clips)
 
#     for clip in predictions:
#         if len(selected_clips) < max_clips:
#             heapq.heappush(selected_clips, (ranking_dict[clip[2]], clip))
#         elif len(goals) > 0 and "goal" not in [c[1][2] for c in selected_clips]:
#             heapq.heappop(selected_clips)
#             heapq.heappush(selected_clips, (ranking_dict[clip[2]], clip))
#         elif ranking_dict[clip[2]] > selected_clips[0][0]:
#             heapq.heappop(selected_clips)
#             heapq.heappush(selected_clips, (ranking_dict[clip[2]], clip))
 
#     selected_clips = [clip[1] for clip in selected_clips]
#     selected_clips.sort(key=lambda x: x[0])  # Sort by time for smooth transitions
 
#     # Stitch clips together using cv2
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = None
#     fps = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))
#     for clip in selected_clips:
#         start, end, label = clip
#         cap = cv2.VideoCapture(video_path)
#         cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
#             if not ret or current_time >= end:
#                 break
 
#             if out is None:
#                 height, width, _ = frame.shape
#                 out = cv2.VideoWriter('highlight_reel.mp4', fourcc, fps, (width, height))
 
#             out.write(frame)
 
#         cap.release()
 
#     if out is not None:
#         out.release()
#         print("Highlight reel generated and saved as highlight_reel.mp4")
#     else:
#         print("No clips selected for the highlight reel.")
 
# # Full execution
# video_path = '/home/kothari.je/video_for_test_highlights/1_720p.mkv'  # Replace with your video path
# predictions = predict_labels(video_path)
# generate_highlight_reel(predictions, video_path)



import os
import cv2
import numpy as np
import torch
from collections import deque
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
    return [index_to_label[p.item()] for p in predictions]
 
# Highlight reel creation logic
def create_highlight_reel(video_clips, output_path, clip_length=20, highlight_duration=300):
    total_clips = highlight_duration // clip_length
    data_structure = deque(maxlen=total_clips)
    all_predictions = []
    
    # print(data_structure)
    # Predict labels for each clip
    for video_clip in video_clips:
        predicted_labels = predict_label(video_clip)
        if predicted_labels:
            all_predictions.append((video_clip, predicted_labels[0]))  # Store clip and its label
    
    # print(all_predictions)
    # Sort clips by time sequence (assuming clips are named sequentially by time)
    all_predictions.sort(key=lambda x: x[0])
 
    # Count goals
    goal_clips = [clip for clip, label in all_predictions if label == "goal"]
 
    # Push clips into the data structure
    ranking_dict = {label: i for i, label in enumerate(label_to_index)}  # Ranking dictionary
    for clip, label in all_predictions:
        if len(data_structure) < total_clips:
            data_structure.append((clip, label))
        elif label == "goal" and "goal" not in [l for _, l in data_structure]:
            # Replace the least important clip
            least_ranked_label = min(data_structure, key=lambda x: ranking_dict[x[1]])
            data_structure.remove(least_ranked_label)
            data_structure.append((clip, label))
 
    # Check if all goals are included
    included_goals = [clip for clip, label in data_structure if label == "goal"]
    if len(included_goals) < len(goal_clips):
        for goal_clip in goal_clips:
            if goal_clip not in included_goals:
                # Replace a less important clip with the missing goal clip
                least_ranked_label = min(data_structure, key=lambda x: ranking_dict[x[1]])
                data_structure.remove(least_ranked_label)
                data_structure.append(goal_clip)
 
    # Stitch clips together
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (224, 224))  # Assuming 224x224 resolution
    print(data_structure)
    for clip, _ in data_structure:
        cap = cv2.VideoCapture(clip)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output_video.write(frame)
        cap.release()
 
    output_video.release()
    print(f"Highlight reel created: {output_path}")
 
# Example Usage
video_clips = [f"/home/kothari.je/test_env/output/{file}" for file in os.listdir("/home/kothari.je/test_env/output") if file.endswith(('.mp4', '.avi'))]
output_path = "highlight_reel.mp4"
create_highlight_reel(video_clips, output_path)
