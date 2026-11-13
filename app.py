import io
import os
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import cv2
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO


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
    "Detect potholes in road images, uploaded videos, and live camera "
    "streams using your fine-tuned YOLO11 model."
)


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"


@st.cache_resource
def load_model(model_path: str):
    """Load the trained YOLO model once and reuse it."""
    return YOLO(model_path)


if not MODEL_PATH.is_file():
    st.error(
        "best.pt was not found. Add your trained best.pt file "
        "to the same GitHub folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the YOLO model: {exc}")
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
# YOLO FRAME DETECTION
# ============================================================

def detect_frame(frame, yolo_model, confidence_threshold, inference_size):
    """Run YOLO on one OpenCV BGR frame and return the annotated frame."""
    results = yolo_model.predict(
        source=frame,
        conf=confidence_threshold,
        imgsz=inference_size,
        verbose=False,
    )

    result = results[0]
    annotated_frame = result.plot()  # Ultralytics returns a BGR image.

    return annotated_frame, result


# ============================================================
# VIDEO PROCESSING WITH H.264 OUTPUT
# ============================================================

def process_video(
    video_bytes,
    filename,
    yolo_model,
    confidence_threshold,
    inference_size,
):
    """Detect potholes frame by frame and encode the result as MP4/H.264."""

    input_path = None
    output_path = None
    input_container = None
    output_container = None

    try:
        suffix = Path(filename).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as input_file:
            input_file.write(video_bytes)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as output_file:
            output_path = output_file.name

        input_container = av.open(input_path)

        video_stream = next(
            (stream for stream in input_container.streams
             if stream.type == "video"),
            None,
        )

        if video_stream is None:
            raise ValueError("The uploaded file does not contain a video stream.")

        rate = (
            video_stream.average_rate
            or video_stream.guessed_rate
            or Fraction(25, 1)
        )
        fps = Fraction(str(float(rate))).limit_denominator(1000)

        width = video_stream.codec_context.width
        height = video_stream.codec_context.height

        # H.264 yuv420p encoding requires even width and height.
        output_width = width - (width % 2)
        output_height = height - (height % 2)

        if output_width < 2 or output_height < 2:
            raise ValueError("The uploaded video has invalid dimensions.")

        output_container = av.open(
            output_path,
            mode="w",
            format="mp4",
        )

        output_stream = output_container.add_stream(
            "libx264",
            rate=fps,
        )
        output_stream.width = output_width
        output_stream.height = output_height
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {
            "crf": "23",
            "preset": "fast",
        }

        total_detections = 0
        frames_with_potholes = 0
        frame_count = 0
        total_frames = video_stream.frames

        progress_bar = st.progress(0)
        status_text = st.empty()

        for frame in input_container.decode(video=0):
            image = frame.to_ndarray(format="bgr24")

            annotated_frame, result = detect_frame(
                image,
                yolo_model,
                confidence_threshold,
                inference_size,
            )

            boxes = result.boxes
            frame_detections = 0 if boxes is None else len(boxes)

            total_detections += frame_detections
            if frame_detections > 0:
                frames_with_potholes += 1

            # Crop to even dimensions for H.264 compatibility.
            annotated_frame = annotated_frame[
                :output_height, :output_width
            ]

            output_frame = av.VideoFrame.from_ndarray(
                annotated_frame,
                format="bgr24",
            )
            output_frame.pts = frame_count
            output_frame.time_base = Fraction(
                fps.denominator, fps.numerator
            )

            for packet in output_stream.encode(output_frame):
                output_container.mux(packet)

            frame_count += 1

            if total_frames > 0:
                progress_bar.progress(
                    min(frame_count / total_frames, 1.0)
                )

            status_text.write(f"Processing frame: {frame_count}")

        # Flush the encoder so the final frames are written.
        for packet in output_stream.encode():
            output_container.mux(packet)

        progress_bar.empty()
        status_text.empty()

        if frame_count == 0:
            raise ValueError("No video frames could be decoded.")

        input_container.close()
        input_container = None

        output_container.close()
        output_container = None

        with open(output_path, "rb") as processed_file:
            processed_video = processed_file.read()

        if not processed_video:
            raise ValueError("The processed video file is empty.")

        return (
            processed_video,
            total_detections,
            frame_count,
            frames_with_potholes,
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
    def __init__(self, yolo_model, confidence_threshold, inference_size):
        self.model = yolo_model
        self.confidence = confidence_threshold
        self.image_size = inference_size

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        annotated_frame, _ = detect_frame(
            image,
            self.model,
            self.confidence,
            self.image_size,
        )

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24",
        )


# ============================================================
# APPLICATION TABS
# ============================================================

image_tab, video_tab, live_tab = st.tabs(
    ["🖼️ Image Detection", "🎥 Video Detection", "📷 Live Camera"]
)


# ============================================================
# IMAGE DETECTION
# ============================================================

with image_tab:
    st.header("Image Pothole Detection")

    uploaded_image = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
    )

    if uploaded_image is not None:
        image = Image.open(
            io.BytesIO(uploaded_image.getvalue())
        ).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        if st.button(
            "🔍 Detect Potholes",
            type="primary",
            key="image_detect",
        ):
            try:
                with st.spinner("Detecting potholes..."):
                    results = model.predict(
                        source=image,
                        conf=confidence,
                        imgsz=image_size,
                        verbose=False,
                    )

                result = results[0]
                annotated_bgr = result.plot()
                annotated_image = Image.fromarray(
                    cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                )

                with right:
                    st.subheader("Detection Result")
                    st.image(
                        annotated_image,
                        use_container_width=True,
                    )

                boxes = result.boxes
                detection_count = 0 if boxes is None else len(boxes)

                st.divider()
                st.subheader("Detection Summary")

                metric1, metric2 = st.columns(2)
                metric1.metric("Potholes Detected", detection_count)
                metric2.metric(
                    "Confidence Threshold",
                    f"{confidence:.0%}",
                )

                if detection_count == 0:
                    st.warning(
                        "No potholes detected. Try lowering the confidence threshold."
                    )
                else:
                    rows = []
                    for box in boxes:
                        class_id = int(box.cls.item())
                        score = float(box.conf.item())
                        x1, y1, x2, y2 = [
                            round(float(value), 1)
                            for value in box.xyxy[0].tolist()
                        ]

                        rows.append(
                            {
                                "Class": result.names.get(
                                    class_id, str(class_id)
                                ),
                                "Confidence": f"{score:.1%}",
                                "X1": x1,
                                "Y1": y1,
                                "X2": x2,
                                "Y2": y2,
                            }
                        )

                    st.dataframe(
                        rows,
                        use_container_width=True,
                        hide_index=True,
                    )

                output = io.BytesIO()
                annotated_image.save(output, format="PNG")

                st.download_button(
                    "⬇️ Download Detected Image",
                    data=output.getvalue(),
                    file_name="pothole_detection_result.png",
                    mime="image/png",
                )

            except Exception as exc:
                st.error(f"Image detection failed: {exc}")


# ============================================================
# UPLOADED VIDEO DETECTION
# ============================================================

with video_tab:
    st.header("Road Video Pothole Detection")

    uploaded_video = st.file_uploader(
        "Upload a road video",
        type=["mp4", "avi", "mov"],
        key="video_upload",
    )

    if uploaded_video is not None:
        st.subheader("Original Video")
        st.video(uploaded_video.getvalue())

        if st.button(
            "🚧 Detect Potholes in Video",
            type="primary",
            key="video_detect",
        ):
            try:
                with st.spinner("Processing video frame by frame..."):
                    (
                        processed_video,
                        detection_count,
                        frames_processed,
                        frames_with_potholes,
                    ) = process_video(
                        uploaded_video.getvalue(),
                        uploaded_video.name,
                        model,
                        confidence,
                        image_size,
                    )

                st.success("Video processing completed!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Frames Processed", frames_processed)
                col2.metric("Total Detections", detection_count)
                col3.metric(
                    "Frames with Potholes",
                    frames_with_potholes,
                )

                st.divider()
                st.subheader("Processed Video")
                st.video(processed_video, format="video/mp4")

                st.download_button(
                    "⬇️ Download Processed Video",
                    data=processed_video,
                    file_name="pothole_detection_video.mp4",
                    mime="video/mp4",
                )

            except Exception as exc:
                st.error(f"Video processing failed: {exc}")


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
                image_size,
            ),
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            async_processing=True,
        )
    except Exception as exc:
        st.error(f"Live camera error: {exc}")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Pothole Detection using fine-tuned YOLO11, "
    "OpenCV, PyAV, Streamlit and Streamlit-WebRTC."
)
