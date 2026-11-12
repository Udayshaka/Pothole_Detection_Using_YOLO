
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("best.pt")

# MediaPipe image API
def process_frame(frame):

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap frame as a MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Get frame data back for YOLO
    processed_frame = mp_image.numpy_view()

    # YOLO pothole detection
    results = model.predict(
        source=processed_frame,
        conf=0.70,
        verbose=False
    )

    # Get annotated frame
    output = results[0].plot()

    return output, results[0]


def process_video(video_path, output_path):

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    total_potholes = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Process each frame
        annotated_frame, result = process_frame(frame)

        # Count potholes
        total_potholes += len(result.boxes)

        out.write(annotated_frame)

    cap.release()
    out.release()

    return total_potholes
