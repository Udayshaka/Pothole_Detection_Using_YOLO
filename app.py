
import io
import os
import tempfile
from pathlib import Path
from fractions import Fraction

import cv2
import av
import numpy as np
import mediapipe as mp
import streamlit as st

from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pothole Detection using YOLO",
    page_icon="🛣️",
    layout="wide",
)

st.title("🛣️ Pothole Detection using YOLO")
st.write(
    "Detect potholes in road images, uploaded videos, "
    "and live camera streams using YOLO11."
)


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"


@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


if not MODEL_PATH.is_file():
    st.error(
        "best.pt not found. Place your trained YOLO weights "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.header("Model Settings")

    st.success("YOLO11 model loaded automatically")

    st.caption("Model: best.pt")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.70,
        step=0.05,
    )

    image_size = st.select_slider(
        "Inference Image Size",
        options=[320, 416, 512, 640, 768, 1024],
        value=640,
    )


# ============================================================
# MEDIAPIPE + YOLO FRAME DETECTION
# ============================================================

def detect_frame(frame, model, confidence, image_size):

    # Convert OpenCV BGR frame to RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Ensure contiguous RGB array
    frame_rgb = np.ascontiguousarray(frame_rgb)

    # Convert frame into MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Get image data from MediaPipe
    image_array = mp_image.numpy_view()

    # YOLO pothole detection
    results = model.predict(
        source=image_array,
        conf=confidence,
        imgsz=image_size,
        verbose=False
    )

    result = results[0]

    # Draw bounding boxes
    annotated_frame = result.plot()

    return annotated_frame, result


# ============================================================
# VIDEO PROCESSING USING PYAV + H264
# ============================================================

def process_video(
    video_bytes,
    filename,
    model,
    confidence,
    image_size
):

    input_path = None
    output_path = None

    input_container = None
    output_container = None

    try:

        # Save uploaded video temporarily
        suffix = Path(filename).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as input_file:

            input_file.write(video_bytes)
            input_path = input_file.name

        # Create output video file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as output_file:

            output_path = output_file.name

        # Open input video
        input_container = av.open(input_path)

        video_stream = next(
            (
                stream
                for stream in input_container.streams
                if stream.type == "video"
            ),
            None
        )

        if video_stream is None:
            raise ValueError("No video stream found.")

        # Get video properties
        fps_value = (
            video_stream.average_rate
            or video_stream.guessed_rate
            or 25
        )

        fps = Fraction(
            str(float(fps_value))
        ).limit_denominator(1000)

        width = video_stream.codec_context.width
        height = video_stream.codec_context.height

        # H264 yuv420p requires even dimensions
        output_width = width - (width % 2)
        output_height = height - (height % 2)

        if output_width <= 0 or output_height <= 0:
            raise ValueError("Invalid video dimensions.")

        # Create output container
        output_container = av.open(
            output_path,
            mode="w",
            format="mp4"
        )

        # Create H264 encoder
        output_stream = output_container.add_stream(
            "libx264",
            rate=fps
        )

        output_stream.width = output_width
        output_stream.height = output_height
        output_stream.pix_fmt = "yuv420p"

        output_stream.options = {
            "crf": "23",
            "preset": "fast"
        }

        output_stream.time_base = Fraction(
            fps.denominator,
            fps.numerator
        )

        # Detection statistics
        total_detections = 0
        frames_with_potholes = 0
        frame_count = 0

        total_frames = video_stream.frames

        progress = st.progress(0)
        status = st.empty()

        # Process video frame by frame
        for frame in input_container.decode(video=0):

            # Convert video frame to OpenCV format
            img = frame.to_ndarray(format="bgr24")

            # Run MediaPipe + YOLO
            annotated_frame, result = detect_frame(
                img,
                model,
                confidence,
                image_size
            )

            # Count detections
            frame_detections = (
                len(result.boxes)
                if result.boxes is not None
                else 0
            )

            total_detections += frame_detections

            if frame_detections > 0:
                frames_with_potholes += 1

            # Crop to even dimensions for H264
            annotated_frame = annotated_frame[
                :output_height,
                :output_width
            ]

            # Convert NumPy array to PyAV frame
            output_frame = av.VideoFrame.from_ndarray(
                annotated_frame,
                format="bgr24"
            )

            # Set video timestamp
            output_frame.pts = frame_count
            output_frame.time_base = Fraction(
                fps.denominator,
                fps.numerator
            )

            # Encode frame
            for packet in output_stream.encode(output_frame):
                output_container.mux(packet)

            frame_count += 1

            # Update progress
            if total_frames > 0:
                progress.progress(
                    min(frame_count / total_frames, 1.0)
                )

            status.write(
                f"Processing frame: {frame_count}"
            )

        # Flush remaining encoder packets
        for packet in output_stream.encode():
            output_container.mux(packet)

        progress.empty()
        status.empty()

        if frame_count == 0:
            raise ValueError("No frames were processed.")

        # Close containers before reading output
        input_container.close()
        input_container = None

        output_container.close()
        output_container = None

        # Read processed video
        with open(output_path, "rb") as file:
            processed_video = file.read()

        if not processed_video:
            raise ValueError("Generated video is empty.")

        return (
            processed_video,
            total_detections,
            frame_count,
            frames_with_potholes
        )

    finally:

        if input_container is not None:
            input_container.close()

        if output_container is not None:
            output_container.close()

        if input_path and os.path.exists(input_path):
            os.remove(input_path)

        if output_path and os.path.exists(output_path):
            os.remove(output_path)


# ============================================================
# LIVE CAMERA PROCESSOR
# ============================================================

class PotholeProcessor(VideoProcessorBase):

    def __init__(self, model, confidence, image_size):

        self.model = model
        self.confidence = confidence
        self.image_size = image_size

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV
        img = frame.to_ndarray(
            format="bgr24"
        )

        # Run detection
        annotated_frame, result = detect_frame(
            img,
            self.model,
            self.confidence,
            self.image_size
        )

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


# ============================================================
# APPLICATION TABS
# ============================================================

image_tab, video_tab, live_tab = st.tabs(
    [
        "🖼️ Image Detection",
        "🎥 Video Detection",
        "📷 Live Camera"
    ]
)


# ============================================================
# IMAGE DETECTION
# ============================================================

with image_tab:

    st.header("Image Pothole Detection")

    uploaded_image = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload"
    )

    if uploaded_image is not None:

        image = Image.open(
            io.BytesIO(uploaded_image.getvalue())
        ).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.subheader("Original Image")
            st.image(
                image,
                use_container_width=True
            )

        if st.button(
            "🔍 Detect Potholes",
            type="primary",
            key="image_detect"
        ):

            try:

                with st.spinner("Detecting potholes..."):

                    results = model.predict(
                        source=image,
                        conf=confidence,
                        imgsz=image_size,
                        verbose=False
                    )

                result = results[0]

                annotated = result.plot()

                annotated_image = Image.fromarray(
                    cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )
                )

                with right:
                    st.subheader("Detection Result")
                    st.image(
                        annotated_image,
                        use_container_width=True
                    )

                boxes = result.boxes

                count = (
                    0 if boxes is None
                    else len(boxes)
                )

                st.divider()
                st.subheader("Detection Summary")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Potholes Detected",
                    count
                )

                col2.metric(
                    "Confidence Threshold",
                    f"{confidence:.0%}"
                )

                if count == 0:

                    st.warning(
                        "No potholes detected. "
                        "Try lowering the confidence threshold."
                    )

                else:

                    rows = []

                    for box in boxes:

                        class_id = int(box.cls.item())
                        score = float(box.conf.item())

                        x1, y1, x2, y2 = [
                            round(float(v), 1)
                            for v in box.xyxy[0].tolist()
                        ]

                        rows.append({

                            "Class": result.names.get(
                                class_id,
                                str(class_id)
                            ),

                            "Confidence": f"{score:.1%}",

                            "X1": x1,
                            "Y1": y1,
                            "X2": x2,
                            "Y2": y2

                        })

                    st.dataframe(
                        rows,
                        use_container_width=True,
                        hide_index=True
                    )

                # Download annotated image
                output = io.BytesIO()

                annotated_image.save(
                    output,
                    format="PNG"
                )

                st.download_button(
                    "⬇️ Download Detected Image",
                    data=output.getvalue(),
                    file_name="pothole_detection_result.png",
                    mime="image/png"
                )

            except Exception as exc:

                st.error(
                    f"Image detection failed: {exc}"
                )


# ============================================================
# VIDEO DETECTION
# ============================================================

with video_tab:

    st.header("Road Video Pothole Detection")

    uploaded_video = st.file_uploader(
        "Upload a road video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if uploaded_video is not None:

        st.subheader("Original Video")

        st.video(uploaded_video)

        if st.button(
            "🚧 Detect Potholes in Video",
            type="primary",
            key="video_detect"
        ):

            try:

                with st.spinner(
                    "Processing video frame by frame..."
                ):

                    (
                        video_bytes,
                        count,
                        frames,
                        detected_frames
                    ) = process_video(
                        uploaded_video.getvalue(),
                        uploaded_video.name,
                        model,
                        confidence,
                        image_size
                    )

                st.success(
                    "Video processing completed successfully!"
                )

                # Display statistics
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Frames Processed",
                    frames
                )

                col2.metric(
                    "Total Pothole Detections",
                    count
                )

                col3.metric(
                    "Frames Containing Potholes",
                    detected_frames
                )

                # Display processed video
                st.divider()

                st.subheader("Processed Video")

                st.video(
                    video_bytes,
                    format="video/mp4"
                )

                # Download processed video
                st.download_button(
                    "⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name="pothole_detection_video.mp4",
                    mime="video/mp4"
                )

            except Exception as exc:

                st.error(
                    f"Video processing failed: {exc}"
                )


# ============================================================
# LIVE CAMERA DETECTION
# ============================================================

with live_tab:

    st.header("Live Road Pothole Detection")

    st.write(
        "Start your camera and point it towards the road "
        "to detect potholes in real time."
    )

    try:

        webrtc_streamer(

            key="pothole-live-camera",

            video_processor_factory=lambda: PotholeProcessor(
                model,
                confidence,
                image_size
            ),

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True

        )

    except Exception as exc:

        st.error(
            f"Live camera error: {exc}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Pothole Detection using fine-tuned YOLO11, "
    "MediaPipe, OpenCV, PyAV and Streamlit."
)
