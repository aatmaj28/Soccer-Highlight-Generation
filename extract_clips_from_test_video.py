import cv2
import os
 
def divide_video_into_clips(video_path, output_folder, clip_duration=20):
    """
    Divides a full video into fixed-duration clips.
 
    Args:
        video_path (str): Path to the input video.
        output_folder (str): Directory where the clips will be saved.
        clip_duration (int): Duration of each clip in seconds (default is 20 seconds).
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
 
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
 
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # Calculate frames per clip
    frames_per_clip = clip_duration * fps
 
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
 
    clip_number = 0
    while True:
        clip_number += 1
        start_frame = (clip_number - 1) * frames_per_clip
        end_frame = min(clip_number * frames_per_clip, total_frames)
 
        # Stop if all frames have been processed
        if start_frame >= total_frames:
            break
 
        # Set the start frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
 
        # Define the output file name
        clip_filename = os.path.join(output_folder, f"clip_{clip_number:03d}.mp4")
 
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_filename, fourcc, fps, (width, height))
 
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
 
        out.release()
        print(f"Saved: {clip_filename}")
 
    cap.release()
    print(f"Video divided into clips and saved in {output_folder}")
 
# Example Usage
video_path = "/home/kothari.je/test_env/input_videos_and_labels/england_epl/2014-2015/2016-09-10 - 17-00 Arsenal 2 - 1 Southampton/2_720p.mkv"  # Path to the video file
output_folder = "/home/kothari.je/test_env/output/"  # Folder to save the clips
divide_video_into_clips(video_path, output_folder, clip_duration=10)