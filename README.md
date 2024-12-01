
# SoccerNet Highlight  Generator

---

## Project Overview

This project automates the generation of highlights for soccer matches by downloading match videos, identifying key moments, and creating a highlight compilation using deep learning models. The pipeline uses **ResNet** for feature extraction and a custom GRU model with an attention mechanism for classification.

---

## Workflow

### 1. Video Downloading
- **Script**: `video_download.py`
- **Description**: Downloads soccer match videos from the **SoccerNet** server.
- **Input**: Match metadata or URLs provided by SoccerNet.
- **Output**: Raw video files saved in a specified directory.

---

### 2. Key Moment Extraction
- **Script**: `feature_extraction.py`
- **Description**: Extracts timestamps of key events (e.g., goals, fouls, etc.) from the SoccerNet-provided `labels.json` file.
- **Input**: JSON file with annotated key moments.
- **Output**: Cropped video clips corresponding to key moments.

---

### 3. Feature Extraction
- **Script**: Integrated in `resnet_training.py`
- **Description**: Processes video frames using a pre-trained **ResNet-50** model to extract features.
- **Input**: Video clips.
- **Output**: ResNet-encoded feature vectors for each frame.

---

### 4. Model Training
- **Script**: `model_training.py`
- **Description**: Trains a **GRU-based model with attention** on the ResNet-encoded features to classify key moments.
- **Input**: ResNet feature vectors.
- **Output**: A PyTorch model (`final_enhanced_gru_model.pth`) for event classification.

---

### 5. Model Evaluation
- **Script**: `accuracy.py`
- **Description**: Evaluates the trained model's performance using metrics like accuracy, precision, recall, etc.
- **Input**: Validation/test dataset with ResNet features and ground-truth labels.
- **Output**: Evaluation metrics.

---

### 6. Highlight Generation
- **Script**: `highlight_generator.py`
- **Description**: Uses the trained model to classify events in a new match video and compiles a highlight reel.
- **Input**: Match video clips.
- **Output**: A highlight reel (`highlight_reel.mp4`).

---

## How to Run

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd SoccerNet-Highlight-Reel-Generator
```

### Step 2: Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Step 3: Download Videos
Run the video download script:
```bash
python video_download.py --config config.json
```

### Step 4: Extract Key Moments
Use `labels.json` to segment the downloaded videos into key moments.

### Step 5: Train the Model
Train the GRU model using the prepared dataset:
```bash
python model_training.py --data_path <path_to_prepared_dataset>
```

### Step 6: Evaluate the Model
Run `accuracy.py` to test model performance:
```bash
python accuracy.py --model_path final_enhanced_gru_model.pth
```

### Step 7: Generate Highlights
Use the trained model to create a highlight reel:
```bash
python highlight_generator.py --input_path <path_to_video_clips> --output_path highlight_reel.mp4
```

---

## Dependencies
- Python 3.8+
- PyTorch
- OpenCV
- torchvision
- NumPy

---

## Directory Structure
```
.
├── video_download.py
├── model_training.py
├── accuracy.py
├── highlight_generator.py
├── requirements.txt
├── labels.json
└── README.md
```

---

## Contributors
- **Your Name**

---

## License
This project is licensed under the MIT License.
