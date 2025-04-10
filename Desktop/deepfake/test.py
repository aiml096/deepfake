import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

# Load models
image_model_path = 'deepfake_detector.h5'  # Replace with your path
video_model_path = 'deepfake_lstm_model.keras'  # Replace with your path

image_model = load_model(image_model_path)
video_model = load_model(video_model_path)

base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Helper functions
def preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_frames_from_video(video_path, output_size=(224, 224), target_frame_count=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_size)
        frames.append(frame)

    cap.release()
    frames = np.array(frames)

    if len(frames) < target_frame_count:
        padding = [frames[-1]] * (target_frame_count - len(frames))
        frames = np.concatenate((frames, padding), axis=0)
    elif len(frames) > target_frame_count:
        frames = frames[:target_frame_count]

    return frames

def extract_features_from_frames(frames):
    features = []
    for frame in frames:
        x = image.img_to_array(frame)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = feature_extractor.predict(x, verbose=0)
        features.append(feature.flatten())

    return np.array(features)

def predict_image(image_file):
    input_image = preprocess_image(image_file)
    prediction = image_model.predict(input_image)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    confidence = prediction if label == "Fake" else 1 - prediction
    return label, confidence

def predict_video(video_file):
    frames = extract_frames_from_video(video_file)
    features = extract_features_from_frames(frames)
    features = np.expand_dims(features, axis=0)
    prediction = video_model.predict(features, verbose=0)
    label = "Fake" if prediction[0][0] > 0.5 else "Real"
    confidence = prediction[0][0] if label == "Fake" else 1 - prediction[0][0]
    return label, confidence

# Streamlit UI
st.title("Deepfake Detection")

st.sidebar.title("Choose Detection Type")
detection_type = st.sidebar.radio("Select an option:", ("Image", "Video"))

if detection_type == "Image":
    st.header("Deepfake Detection for Images")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect"):
            label, confidence = predict_image(uploaded_image)
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2f}")

elif detection_type == "Video":
    st.header("Deepfake Detection for Videoss")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        if st.button("Detect"):
            with st.spinner("Analyzing video... This may take some time."):
                temp_video_path = "temp_video.mp4"
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_video.read())
                label, confidence = predict_video(temp_video_path)
                os.remove(temp_video_path)
                st.success(f"Prediction: {label}")
                st.info(f"Confidence: {confidence:.2f}")
