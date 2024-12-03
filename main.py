import cv2
import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO

# Define the categories
categories = ['Electronics', 'biological', 'cardboard', 'clothes',
              'glass', 'metal', 'paper', 'plastic', 'shoes']

# Load YOLOv8 Model (cached to avoid reloading every time)
import os

# Use relative path to load the model file
model_path = os.path.join(os.getcwd(), 'best.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the model (example for PyTorch)
import torch
model = torch.load(model_path)

def make_prediction(img, confidence_threshold=0.5):
    img_array = np.array(img)  # Convert PIL image to numpy array
    results = model(img_array)  # Get predictions from YOLOv8
    prediction = results[0]  # Take the first result from the predictions

    # Extract bounding boxes, labels, and confidence scores
    boxes = prediction.boxes.xyxy.int()  # Bounding boxes (converted to integer format)
    labels = [categories[int(cls)] for cls in prediction.boxes.cls]  # Class labels
    scores = prediction.boxes.conf  # Confidence scores

    # Filter based on confidence threshold
    filtered_indices = [i for i, score in enumerate(scores) if score >= confidence_threshold]
    boxes = boxes[filtered_indices]
    labels = [labels[i] for i in filtered_indices]
    scores = [scores[i] for i in filtered_indices]

    return {"boxes": boxes, "labels": labels, "scores": scores}

def create_image_with_bboxes(img, prediction):
    # Ensure the image is in RGB format
    if img.shape[2] == 4:  # Convert RGBA to RGB if needed
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert image to tensor format [C, H, W]
    img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).byte()

    # Draw bounding boxes on the image
    img_with_bboxes = draw_bounding_boxes(
        img_tensor,
        boxes=prediction["boxes"],
        labels=prediction["labels"],
        colors=["red" for _ in prediction["labels"]],
        width=2
    )

    # Convert the tensor back to a numpy array
    img_with_bboxes_np = img_with_bboxes.permute(1, 2, 0).numpy()
    return img_with_bboxes_np

# Streamlit Dashboard
st.title("Object Detector using YOLOv8")


st.markdown("Select either Image Upload or Webcam to detect multiple objects:")


confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Option to upload an image or use the webcam
image_upload_option = st.radio("Choose an option:", ("Image Upload", "Webcam"))

if image_upload_option == "Image Upload":
    uploaded_file = st.file_uploader("Upload Image:", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)

        # Make predictions on the uploaded image with the selected confidence threshold
        prediction = make_prediction(img, confidence_threshold)

        # Add bounding boxes to the image
        img_with_bbox = create_image_with_bboxes(np.array(img), prediction)

        # Display the image with bounding boxes
        st.image(img_with_bbox, channels="RGB", use_column_width=True)

        # Display predictions (labels and confidence scores)
        st.header("Detected Objects")
        for label, score in zip(prediction["labels"], prediction["scores"]):
            st.write(f"Label: {label}, Confidence: {score:.2f}")

elif image_upload_option == "Webcam":
    if 'run' not in st.session_state:
        st.session_state.run = False

    detect_button = st.button("Start Detection")
    stop_button = st.button("Stop Detection")

    if detect_button:
        st.session_state.run = True

    if stop_button:
        st.session_state.run = False

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("Could not access the webcam.")
    else:
        while st.session_state.run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to read from webcam.")
                break

            # Make predictions with the selected confidence threshold
            prediction = make_prediction(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), confidence_threshold)

            # Add bounding boxes
            frame_with_bbox = create_image_with_bboxes(frame, prediction)

            # Display the frame with bounding boxes
            FRAME_WINDOW.image(frame_with_bbox, channels="RGB")

            # Display detected objects
            st.header("Detected Objects")
            for label, score in zip(prediction["labels"], prediction["scores"]):
                st.write(f"Label: {label}, Confidence: {score:.2f}")

    camera.release()
