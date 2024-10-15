import os
from dotenv import load_dotenv
import cv2
from inference import get_model
from video_processing import VideoProcessor
from predictions import BallPredictor

# Load environment variables from .env file
load_dotenv()

# Load a pre-trained YOLOv8 model
model = get_model(model_id="drs-5gvgi/3")

# Define input and output video file paths
input_video_file = os.getenv('INPUT_VIDEO_PATH', "v6_crop.mp4")
output_video_file = os.getenv('OUTPUT_VIDEO_PATH', "output_video2.mp4")

# Initialize the video processor
video_processor = VideoProcessor(model, input_video_file, output_video_file)

# Process the video frames
video_processor.process_frames()

# Predict future ball trajectory and hitting status
ball_predictor = BallPredictor(video_processor.ball_trajectory)
future_path = ball_predictor.predict_future_path(num_future_frames=10, scaling_factor=1.0)
predicted_hitting_stumps = ball_predictor.predict_hitting_stumps(future_path, video_processor.stump_properties)

# Print prediction results
if predicted_hitting_stumps:
    print("The ball is predicted to hit the stumps.")
else:
    print("The ball is predicted NOT to hit the stumps.")
