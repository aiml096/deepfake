# skin_disease_app.py

import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# Load the model
model = load_model("skindisease.h5")

# Disease classes
classes = ['Acne', 'Melanoma', 'Peeling skin', 'Ring worm', 'Vitiligo']

# Streamlit app
st.title("Skin Disease Detection")
st.write("Upload an image of the skin condition to get a prediction.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = img.resize((64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # Prediction
    preds = model.predict(x)
    label = np.argmax(preds, axis=1)[0]
    prediction = classes[label]

    st.success(f"The predicted skin disease is: **{prediction}**")
