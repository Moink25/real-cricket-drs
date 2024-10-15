import os
from dotenv import load_dotenv
import cv2
import numpy as np
from inference import get_model
import supervision as sv

# Load environment variables from .env file
load_dotenv()

# Load a pre-trained YOLOv8 model
model = get_model(model_id="drs-5gvgi/3")

# Define input and output video file paths
input_video_file = "./images/input/video107_crop.mp4"  # Path to your input video file
output_video_file = "./images/output/video107_output.mp4"  # Path to save the annotated output video

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
num_future_frames = 7
scaling_factor = 0.4
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

    # Draw the solid pipe-like trajectory of the ball
    if len(ball_trajectory) > 1:
        for point in ball_trajectory:
            cv2.circle(annotated_frame, point, 8, (0, 0, 255), -1)
        cv2.polylines(annotated_frame, [np.array(ball_trajectory)], False, (255, 0, 0), thickness=4)

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

# Predict the future path at the end
if len(ball_trajectory) > num_points_for_prediction:
    # Calculate speed of the ball
    speeds = []
    for i in range(1, len(ball_trajectory)):
        delta_x = ball_trajectory[i][0] - ball_trajectory[i - 1][0]
        delta_y = ball_trajectory[i][1] - ball_trajectory[i - 1][1]
        speed = np.sqrt(delta_x ** 2 + delta_y ** 2)  # Euclidean distance
        speeds.append(speed)

    # Calculate average speed
    avg_speed = np.mean(speeds)

    # Adjust number of future frames based on speed
    if avg_speed < 10:  # Adjust threshold based on your requirements
        num_future_frames = 5  # More frames for slow speeds
    elif avg_speed < 20:
        num_future_frames = 5  # Medium frames for medium speeds
    else:
        num_future_frames = 5  # Less frames for fast speeds

    total_delta_x = 0
    total_delta_y = 0
    count = 0
    for i in range(-num_points_for_prediction, -1):
        delta_x = ball_trajectory[i + 1][0] - ball_trajectory[i][0]
        delta_y = ball_trajectory[i + 1][1] - ball_trajectory[i][1]
        total_delta_x += delta_x
        total_delta_y += delta_y
        count += 1
    avg_delta_x = (total_delta_x / count) * scaling_factor
    avg_delta_y = (total_delta_y / count) * scaling_factor

    # Calculate the future trajectory points
    future_path = []
    x_last, y_last = ball_trajectory[-1]
    for i in range(1, num_future_frames + 1):
        future_x = int(x_last + i * avg_delta_x)
        future_y = int(y_last + i * avg_delta_y)
        future_path.append((future_x, future_y))

    # Check if the last predicted location is outside the stumps area
    predicted_hitting_stumps = False

    # Check only the last predicted point for hitting the stumps
    last_predicted_point = future_path[-1]
    if stump_x <= last_predicted_point[0] <= stump_x + stump_width and last_predicted_point[1] >= stump_y:
        predicted_hitting_stumps = True

    # Draw the stumps only during future path prediction
    overlay = np.zeros_like(annotated_frame)  # Create a blank overlay for the stumps
    cv2.rectangle(overlay, (stump_x, stump_y), (stump_x + stump_width, height), (0, 255, 0), -1)  # Green stumps
    alpha = 0.5  # Transparency level (0.0 to 1.0)

    # Draw the future path on the last frame
    last_frame = annotated_frame.copy()  # Use the last annotated frame for prediction
    last_frame = cv2.addWeighted(overlay, alpha, last_frame, 1 - alpha, 0)  # Blend the stumps overlay

    # Draw the future path
    for point in future_path:
        cv2.circle(last_frame, point, 8, (0, 255, 255), -1)  # Yellow color for future points
    cv2.polylines(last_frame, [np.array(future_path)], False, (0, 255, 255), thickness=4)

    # Display prediction message
    if predicted_hitting_stumps:
        prediction_message = "Predicted Ball Hitting Stumps!"
    else:
        prediction_message = "Predicted Ball Not Hitting Stumps"

    cv2.putText(last_frame, prediction_message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Write the last frame with predictions to the output video
    out.write(last_frame)

# Pause for a few seconds for result analysis
for _ in range(int(fps * 3)):  # 3 seconds pause
    out.write(last_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
