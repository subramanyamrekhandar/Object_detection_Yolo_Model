import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2
from ultralytics import YOLO
from streamlit_option_menu import option_menu
from pytube import YouTube

# Paths to models and class files
IMAGE_MODEL_PATH = "Yolo_model_9d/best.pt"  # Model for image detection
VIDEO_MODEL_PATH = "Models/yolov8n.pt"  # Model for video detection
CLASS_FILE = "items_dataset9.txt"  # Classes for image model
COCO_CLASS_FILE = "coco.names"  # Classes for video model

# Load YOLO models
image_model = YOLO(IMAGE_MODEL_PATH)
video_model = YOLO(VIDEO_MODEL_PATH)

# Load class lists
with open(CLASS_FILE, "r") as f:
    image_class_list = [line.strip() for line in f if line.strip()]

with open(COCO_CLASS_FILE, "r") as f:
    video_class_list = f.read().splitlines()

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Image Detection", "Video Detection", "YouTube Video Detection", "About"],
        icons=["image", "camera-video", "youtube", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Function for image detection
def detect_objects_in_image(image_path):
    results = image_model(image_path)
    annotated_image = results[0].plot()
    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls.item())
        class_name = results[0].names[class_id]
        if class_name in image_class_list:
            detections.append({
                "Class": class_name,
                "Confidence": round(float(box.conf.item()), 2),
                "X_min": round(float(box.xyxy[0][0].item()), 2),
                "Y_min": round(float(box.xyxy[0][1].item()), 2),
                "X_max": round(float(box.xyxy[0][2].item()), 2),
                "Y_max": round(float(box.xyxy[0][3].item()), 2),
            })
    return annotated_image, detections

# Function for video detection with display
def detect_objects_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Streamlit video placeholder
    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = video_model(frame, stream=True)

        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            class_ids = result.boxes.cls

            for box, conf, cls in zip(boxes, confs, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = video_class_list[int(cls)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name} ({conf:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert frame to RGB (Streamlit uses RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update video placeholder with processed frame
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Delay for smooth playback
        cv2.waitKey(int(1000 / fps)) 

    cap.release()
    cv2.destroyAllWindows()

# Function to process YouTube video
def process_youtube_video(youtube_url):
    try:
        st.write("Downloading YouTube video...")
        yt = YouTube(youtube_url)
        stream = yt.streams.get_highest_resolution()
        video_path = os.path.join("uploads", "youtube_video.mp4")
        os.makedirs("uploads", exist_ok=True)
        stream.download(filename=video_path)
        st.success("YouTube video downloaded successfully!")
        return video_path
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}")
        return None

# Image Detection Page
if selected == "Image Detection":
    st.title("Object Detection in Images")
    st.write("Upload an image to detect objects.")

    upload = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if upload:
        image_path = os.path.join("uploads", upload.name)
        os.makedirs("uploads", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(upload.getbuffer())
        img = Image.open(image_path)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        annotated_image, detections = detect_objects_in_image(image_path)
        st.image(annotated_image, caption="Detected Objects", use_column_width=True)
        if detections:
            st.write("### Detected Objects")
            st.dataframe(detections)
        else:
            st.write("No objects detected.")

# Video Detection Page
elif selected == "Video Detection":
    st.title("Object Detection in Videos")
    st.write("Upload a video to detect objects in real-time.")
    
    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if video_file:
        video_path = os.path.join("uploads", video_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())

        st.write("### Playing Video with Object Detection...")
        detect_objects_in_video(video_path)

# YouTube Video Detection Page
elif selected == "YouTube Video Detection":
    st.title("Object Detection in YouTube Videos")
    st.write("Enter a YouTube video link to analyze objects in the video.")

    youtube_url = st.text_input("Enter YouTube Video URL:")
    if st.button("Process YouTube Video"):
        if youtube_url:
            video_path = process_youtube_video(youtube_url)
            if video_path:
                st.write("### Playing YouTube Video with Object Detection...")
                detect_objects_in_video(video_path)
        else:
            st.error("Please enter a valid YouTube video URL.")

# About Page
elif selected == "About":
    st.title("About")
    st.write("""
    This application detects objects in images, uploaded videos, and YouTube videos using YOLO models.
    - **Image Detection**: Uses a custom YOLO model.
    - **Video Detection**: Uses YOLOv8 trained on COCO dataset.
    - **YouTube Video Detection**: Downloads YouTube videos and detects objects in them.
    
    Created by [Subramanyam Rekhandar](https://www.linkedin.com/in/subramanyamrekhandar/).
    """)
