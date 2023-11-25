import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize RealSense Pipeline
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe.start(cfg)
depth_image_colormap = np.zeros((480, 640, 3), dtype=np.uint8)

# Create a new window for displaying pose, CoG, and quadrilateral
cv2.namedWindow("Pose and CoG", cv2.WINDOW_NORMAL)

# Initialize variables for posture accumulation
posture_accumulation = []
start_time = time.time()
accumulation_interval = 2  # seconds
display_duration = 5  # seconds
last_majority_posture = None
last_majority_start_time = 0

# Initialize feet_points outside the loop
feet_points = []

while True:
    # RealSense Camera
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # MediaPipe Pose on RGB frames
    imgRGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # Initialize status outside the if block
    status = ""

    if results.pose_landmarks:
        # Extract landmarks for key points
        landmarks = results.pose_landmarks.landmark
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

        # Calculate the angle between the hips and knees
        angle_hip_knee_left = abs(left_hip.y - left_knee.y)
        angle_hip_knee_right = abs(right_hip.y - right_knee.y)

        # Set a threshold for sitting detection
        sit_threshold = 0.1  # You may need to adjust this based on your specific scenario

        # Check if the person is sitting
        if angle_hip_knee_left < sit_threshold and angle_hip_knee_right < sit_threshold:
            status = "Sitting"
        else:
            status = "Standing"

        # Record the posture for accumulation
        posture_accumulation.append(status)

        # Draw landmarks on the image
        mp_drawing.draw_landmarks(color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(color_image, f"Status: {status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Get landmarks for the feet (landmarks 29, 31, 30, and 32)
        feet_landmarks = [landmarks[29], landmarks[30], landmarks[32], landmarks[31]]

        # Update feet_points outside the loop
        feet_points = [(int(landmark.x * color_image.shape[1]), int(landmark.y * color_image.shape[0])) for landmark in feet_landmarks]

        # Calculate Center of Gravity (CoG) based on all landmarks
        cog_x = int(np.mean([lm.x for lm in landmarks]) * color_image.shape[1])
        cog_y = int(np.mean([lm.y for lm in feet_landmarks]) * color_image.shape[0])

    # Display the depth image with pose landmarks and the quadrilateral
    cv2.imshow("RGB Image with Pose Landmarks", color_image)

    # Create a new image for displaying pose, CoG, and quadrilateral
    pose_cog_quad_img = np.zeros((480, 640, 3), dtype=np.uint8)

    if feet_points:
        # Draw pose landmarks on the new image
        mp_drawing.draw_landmarks(pose_cog_quad_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw the quadrilateral on the new image
        cv2.polylines(pose_cog_quad_img, [np.array(feet_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw a marker at the CoG position on the new image
        cv2.circle(pose_cog_quad_img, (cog_x, cog_y), 10, (255, 255, 255), -1)

    # Display the new image with pose, CoG, and quadrilateral
    cv2.imshow("Pose and CoG", pose_cog_quad_img)

    # Check if accumulation interval is reached
    elapsed_time = time.time() - start_time
    if elapsed_time >= accumulation_interval:
        # Check if the posture accumulation list is not empty
        if posture_accumulation:
            # Determine the majority posture
            majority_posture = max(set(posture_accumulation), key=posture_accumulation.count)

            # Check if the majority posture has changed
            if majority_posture != last_majority_posture:
                last_majority_posture = majority_posture
                last_majority_start_time = time.time()

            # Check if the display duration has passed since the last majority change
            elapsed_display_time = time.time() - last_majority_start_time
            if elapsed_display_time <= display_duration:
                # Display the majority posture
                cv2.putText(color_image, f"Majority: {last_majority_posture}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Reset accumulation variables
                posture_accumulation = []
                start_time = time.time()

    # Handle keyboard input
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Cleanup
pipe.stop()
cv2.destroyAllWindows()
