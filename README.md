# **SoccerNet Highlight Generator**

## **Overview**
This project automates the generation of soccer match highlights by downloading match videos, identifying key moments, and creating highlight compilations. It uses **ResNet-50** for feature extraction and a custom **GRU model with an attention mechanism** for classification. For a live demonstration, the model can be deployed using tools like Gradio for interactive exploration.

---

## **Features**

### **Data Preparation**
- **Dataset Source**: Utilizes the SoccerNet dataset, which includes full soccer match videos and JSON files with detailed annotations of key events.
- **Clip Extraction**: Parses the JSON annotations to extract 20-second video clips around each key moment. For example, if a goal occurs at 20 minutes, the script extracts a clip from 19:50 to 20:10 and saves it in the respective **goals** folder.
- **Organized Training Data**: Creates labeled folders (e.g., `goals`, `fouls`, `corners`) containing 20-second clips of corresponding actions.

---

### **Model Training**
- **Feature Extraction**: Converts 20-second video clips into tensors using a pre-trained **ResNet-50** model, extracting frame-level feature vectors.
- **Training Process**: Pairs the ResNet-encoded tensors with their respective labels (from the folder names) and feeds them into a custom **GRU model with attention**.
- **Fine-Tuning**: Fine-tunes the GRU model to accurately classify the type of action happening in each clip (e.g., goal, foul, corner).
- **Output**: Produces a trained model capable of identifying soccer actions from 20-second video clips.

---

### **Action Detection**
- **Video Segmentation**: Splits a full soccer match video into 20-second segments.
- **Action Classification**: Passes each segment through the trained model, which identifies the type of action occurring in the clip (e.g., goal, foul, kickoff).
- **Action Ranking**: Assigns importance scores to actions based on pre-defined rankings to prioritize key moments.

---

### **Highlight Generation**
- **Key Moment Selection**: Selects the most critical moments from the identified actions based on their rankings.
- **Highlight Compilation**: Stitches together the selected clips into a smooth and compact highlight reel, showcasing the most exciting moments of the match.
- **Automated Workflow**: Provides users with a fully automated process that converts a full soccer match into an engaging highlight video.

---

### **User Interaction**
- **Input**: Users upload a full soccer match video to the platform.
- **Output**: The system processes the video and generates a polished highlight reel featuring all the top moments, ready for review or sharing.

---

## **Document Overview**
- `video_download.py`: Script to download soccer match videos from the SoccerNet server.
- `label_extraction.py`: Extracts key timestamps from annotated `labels.json` files.
- `resnet50_feature_extraction.py`: Integrates ResNet feature extraction into the pipeline.
- `model_training.py`: Trains the GRU-based model with attention for event classification.
- `inference_on_all_videos.py`: Evaluates model performance using metrics like accuracy, precision, and recall.
- `test_video_to_clips.py`: Divides a full soccer match videos into 10 sec clips for prediction and stores it.
- `highlight_generator.py`: Creates a highlight reel by classifying and compiling video clips.

---

## **Getting Started**

### **Prerequisites**
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- OpenCV
- torchvision
- NumPy

### **Installation**

### Step 1: Clone the Repository
```bash
git clone https://github.com/aatmaj28/Soccer-Highlight-Generation.git
cd Soccer-Highlight-Generation
```

### Step 2: Install Dependencies
Install the required Python packages:
```bash
pip install <all packages mentioned above>
```

### Step 3: Download Videos
Run the video download script:
```bash
python video_download.py
```

### Step 4: Extract Key Moments from videos into clips
```bash
python label_extraction.py
```

### Step 5: Extract Features into feature vector
```bash
python resnet50_feature_extraction.py
```

### Step 6: Train the Model
Train the GRU model using the prepared dataset:
```bash
python model_training.py
```

### Step 7: Evaluate the Model
Run `inference_on_all_videos.py` to test model performance:
```bash
python inference_on_all_videos.py
```

### Step 8: Generate Highlights
Use the trained model to create a highlight reel:
```bash
python test_video_to_clips.py
python highlight_generation.py
```

## **Contributing**
Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request.

---

## **Acknowledgments**
This project utilizes the following resources and libraries:
- [PyTorch](https://pytorch.org/) for deep learning models.
- [OpenCV](https://opencv.org/) for video processing.
- [SoccerNet](https://www.soccer-net.org/) for providing video data and annotations.
- Pre-trained **ResNet-50** for feature extraction.

---

## **Resources**
The models were trained using and combining the following resources:
- [SoccerNet datasets](https://www.soccer-net.org/) for annotated match videos.
- Custom-built datasets with enhanced event annotations.
