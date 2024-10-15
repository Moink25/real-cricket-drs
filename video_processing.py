# video_processing.py

import os
from dotenv import load_dotenv
import cv2
import numpy as np
from inference import get_model
import supervision as sv
from bail_dislodgment import BailDislodgment  # Import the class

# Load environment variables from .env file
load_dotenv()

# Load a pre-trained YOLOv8 model
model = get_model(model_id="drs-5gvgi/3")

# Define input and output video file paths
input_video_file = "v6_crop.mp4"  # Path to your input video file
output_video_file = "output_video2.mp4"  # Path to save the annotated output video

# Open the input video
cap = cv2.VideoCapture(input_video_file)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the stump area properties as fractions of the width and height
stump_width = int(0.022 * width)  # 2.5% of the video width
stump_height = int(0.74 * height)  # 75% of the video height
stump_x = int(width / 2 - stump_width / 2)  # Center x-coordinate of the stumps
stump_y = height - stump_height  # y-coordinate for the bottom of the stumps

# Define tracking boundaries for the ball
left_bound = 0  # Left boundary (0 pixels from the left edge)
right_bound = width  # Right boundary (width of the video frame)

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_file, fourcc, fps * 0.3, (width, height))

# Create Supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Initialize list to track the ball's trajectory
ball_trajectory = []

# Number of future frames to predict at the end
num_future_frames = 8
scaling_factor = 0.5
num_points_for_prediction = 5

# Process the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the current frame
    results = model.infer(frame)[0]

    # Load the results into the Supervision Detections API
    detections = sv.Detections.from_inference(results)

    # Annotate the frame with inference results
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Extract the ball's bounding box and track its position within the defined boundaries
    ball_hitting = False  # Variable to track if the ball is hitting the stumps
    ball_passed_stumps = False  # Variable to track if the ball has passed the stumps area
    ball_velocity = 0  # Placeholder for ball velocity

    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = detections.xyxy[i]
        class_id = detections.class_id[i]
        if class_id == 0:  # Assuming class_id 0 corresponds to the ball
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            if left_bound <= x_center <= right_bound:
                ball_trajectory.append((x_center, y_center))
                cv2.circle(annotated_frame, (x_center, y_center), 5, (0, 255, 0), -1)

                # Check for collision with the stumps
                if stump_x <= x_center <= stump_x + stump_width and y_center >= stump_y:
                    ball_hitting = True  # Ball is hitting the stumps
                elif y_center < stump_y and x_center > stump_x + stump_width:
                    ball_passed_stumps = True  # Ball passed the stumps without hitting

                # Calculate ball velocity
                if len(ball_trajectory) > 1:
                    delta_x = ball_trajectory[-1][0] - ball_trajectory[-2][0]
                    delta_y = ball_trajectory[-1][1] - ball_trajectory[-2][1]
                    ball_velocity = np.sqrt(delta_x ** 2 + delta_y ** 2)  # Euclidean distance

    # Check dislodgment only if ball is hitting the stumps
    bail_dislodged = False
    if ball_hitting:
        mass_ball = 0.155  # Mass of the cricket ball in kg
        mass_bail = 0.05  # Mass of the bail in kg
        height_bail = 0.15  # Height of the bail from the ground in meters
        angle_impact = 45  # Angle of impact, can be adjusted as needed

        # Create BailDislodgment object and check if the bail is dislodged
        bail_checker = BailDislodgment(mass_ball, ball_velocity, angle_impact, mass_bail, height_bail)
        bail_dislodged = bail_checker.is_bail_dislodged()
        if bail_dislodged:
            cv2.putText(annotated_frame, "Bail Dislodged!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw the solid pipe-like trajectory of the ball
    if len(ball_trajectory) > 1:
        for point in ball_trajectory:
            cv2.circle(annotated_frame, point, 8, (0, 0, 255), -1)
        cv2.polylines(annotated_frame, [np.array(ball_trajectory)], False, (255, 0, 0), thickness=4)

    # Predict future path of the ball
    if len(ball_trajectory) >= num_points_for_prediction:
        future_trajectory = []
        last_point = ball_trajectory[-1]
        for i in range(1, num_future_frames + 1):
            future_x = int(last_point[0] + (ball_velocity * scaling_factor * i))
            future_y = int(last_point[1] + (ball_velocity * scaling_factor * i))
            future_trajectory.append((future_x, future_y))

        # Draw future path
        for point in future_trajectory:
            cv2.circle(annotated_frame, point, 5, (255, 255, 0), -1)
        if len(future_trajectory) > 1:
            cv2.polylines(annotated_frame, [np.array(future_trajectory)], False, (0, 255, 255), thickness=4)

    # Display message on the frame
    if ball_hitting:
        message = "Ball Hitting Stumps!"
        cv2.putText(annotated_frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    elif ball_passed_stumps:
        message = "Ball Passed Stumps!"
        cv2.putText(annotated_frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        message = "Ball Not Hitting Stumps"
        cv2.putText(annotated_frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# After processing all frames, pause for analysis
for _ in range(3 * int(fps)):  # Pause for 3 seconds
    # Create a blank frame for displaying the analysis message
    analysis_frame = np.zeros((height, width, 3), dtype=np.uint8)
    analysis_text = "Analysis: Ball " + ("HIT" if ball_hitting else "MISSED") + " the Stumps"
    cv2.putText(analysis_frame, analysis_text, (50, height // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    out.write(analysis_frame)  # Write the analysis frame to the output video

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
