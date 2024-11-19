import os
import streamlit as st
import torch
import cv2
import numpy as np
import torch.nn as nn
import torchvision.models as models
import tempfile
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.eval() 
resnet.load_state_dict(torch.load('/Users/devadarshini/Desktop/resnet_model.pth', weights_only=True))

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
gru_model = GRUModel(input_size=1000, hidden_size=512, output_size=2) 
gru_model.load_state_dict(torch.load('/Users/devadarshini/Desktop/gru_model.pth', weights_only=True))
gru_model.eval() 
st.title('Highlight Generation')
st.markdown('Upload a video to extract frames and classify events.')

def display_frames(frames):
    for frame_path in frames:
        img = cv2.imread(frame_path)
        st.image(img, channels="BGR")

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int((total_frames / fps) * 1000)

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224)) 
        frame = frame.astype(np.float32) / 255.0 
        input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            feature = resnet(input_tensor) 
            features.append(feature.view(-1).numpy())  

    cap.release()
    features = np.array(features)
    features = features.reshape(1, features.shape[0], -1) 
    return features

def extract_frames_in_timeframe(video_path, start_time, end_time, output_dir, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return []

    # Get video duration
    total_duration = get_video_duration(video_path)
    if total_duration == 0:
        return []

    # Adjust start and end times if necessary
    if start_time >= total_duration:
        st.warning("Start time exceeds video duration. Adjusting to start of video.")
        start_time = 0
    if end_time > total_duration:
        st.warning("End time exceeds video duration. Adjusting to end of video.")
        end_time = total_duration
    if start_time >= end_time:
        st.error("Invalid time range: Start time must be less than end time.")
        return []

    # Set video position and extract frames
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time)
    frames = []
    frame_count = 0
    total_frames_to_extract = int((end_time - start_time) * frame_rate / 1000)

    while frame_count < total_frames_to_extract:
        success, frame = cap.read()
        if not success:
            break
        frame_file = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if cv2.imwrite(frame_file, frame):
            frames.append(frame_file)
        frame_count += 1

    cap.release()
    return frames

def create_video_from_frames(frame_files, output_video_path, frame_rate=30):
    if not frame_files:
        return

    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()

if uploaded_video := st.file_uploader("Choose a video file", type=["mp4", "mkv"]):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video.name.split('.')[-1]}") as temp_video_file:
        temp_video_file.write(uploaded_video.read())
        video_path = temp_video_file.name

    st.video(video_path)
    st.write("Extracting features and making predictions...")

    video_features = extract_features_from_video(video_path)
    video_features_tensor = torch.tensor(video_features, dtype=torch.float32)

    with torch.no_grad():
        outputs = gru_model(video_features_tensor)
        _, predicted = torch.max(outputs, 1)

    label_mapping = ['Kick-off', 'Ball out of play']
    predicted_label = label_mapping[predicted.item()]
    st.write(f"Predicted label: {predicted_label}")

    total_duration = get_video_duration(video_path)
    st.write(f"Video duration: {total_duration // 1000} seconds.")

    start_time = st.slider("Select start time (ms)", min_value=0, max_value=total_duration, value=50000)
    end_time = st.slider("Select end time (ms)", min_value=0, max_value=total_duration, value=60000)

    frame_rate = st.slider("Select frame rate", min_value=1, max_value=60, value=30)

    with tempfile.TemporaryDirectory() as temp_dir:
        frames = extract_frames_in_timeframe(video_path, start_time, end_time, temp_dir, frame_rate)
        if frames:
            st.write(f"Displaying {len(frames)} extracted frames:")
            display_frames(frames)

            if st.button("Create Video from Frames"):
                output_video_path = os.path.join(temp_dir, 'extracted_video.mp4')
                create_video_from_frames(frames, output_video_path)
                st.write(f"Video created: {output_video_path}")
                st.video(output_video_path)

                st.download_button("Download Video", output_video_path, file_name="extracted_video.mp4")

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(video_features.reshape(-1, 1)) 
    st.write("Visualizing extracted features using PCA:")
    st.pyplot()
    if 'y_true' in locals() and 'y_pred' in locals():
        st.text(classification_report(y_true, y_pred))  
