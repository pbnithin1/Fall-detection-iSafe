import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
import numpy as np

# Initialize MediaPipe Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Initialize RealSense Pipeline
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe.start(cfg)
depth_image_colormap = np.zeros((480, 640, 3), dtype=np.uint8)

pTime = 0

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

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

       # Convert depth image to BGR for drawing landmarks
        depth_image_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
        mpDraw.draw_landmarks(depth_image_colormap, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Get landmarks for the feet (landmarks 29, 31, 30, and 32)
        feet_landmarks = [landmarks[29], landmarks[30], landmarks[32], landmarks[31]]

        # Draw a quadrilateral using the feet landmarks
        feet_points = [(int(landmark.x * color_image.shape[1]), int(landmark.y * color_image.shape[0])) for landmark in feet_landmarks]
        cv2.polylines(depth_image_colormap, [np.array(feet_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Calculate the area of the quadrilateral using the Shoelace formula
        area = 0.5 * abs(
            (feet_landmarks[0].x * (feet_landmarks[1].y - feet_landmarks[2].y) +
             feet_landmarks[1].x * (feet_landmarks[2].y - feet_landmarks[0].y) +
             feet_landmarks[2].x * (feet_landmarks[0].y - feet_landmarks[1].y) +
             feet_landmarks[0].x * (feet_landmarks[1].y - feet_landmarks[3].y) +
             feet_landmarks[1].x * (feet_landmarks[3].y - feet_landmarks[0].y) +
             feet_landmarks[3].x * (feet_landmarks[0].y - feet_landmarks[1].y))
        )

        # Calculate Center of Gravity (CoG) based on all landmarks
        cog_x = int(np.mean([lm.x for lm in landmarks]) * color_image.shape[1])
        cog_y = int(np.mean([lm.y for lm in feet_landmarks]) * color_image.shape[0])

        # Draw a marker at the CoG position
        cv2.circle(depth_image_colormap, (cog_x, cog_y), 10, (255, 255, 255), -1)

        # Check if the distance between CoG and the center of the quadrilateral is too large
        quad_center = (sum([p[0] for p in feet_points]) // 4, sum([p[1] for p in feet_points]) // 4)
        distance = np.sqrt((cog_x - quad_center[0]) ** 2 + (cog_y - quad_center[1]) ** 2)

        if distance > 50:  # Adjust this threshold as needed
            cv2.putText(color_image, "FALL IMMINENT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(color_image, f"FPS: {int(fps)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Display the depth image with pose landmarks and the quadrilateral
    cv2.imshow("Depth Image with Pose Landmarks", depth_image_colormap)
    cv2.imshow("RGB Image with Pose Landmarks", color_image)

    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
pipe.stop()
cv2.destroyAllWindows()
