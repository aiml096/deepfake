import streamlit as st
from PIL import Image
import io
import os

# UI setup
st.set_page_config(page_title="AgriDerm", layout="centered")
st.title("🌾 AgriDerm: Skin Disease Detection for Farmers")
st.write("Upload a skin image **or use your camera** to get the image file name (without extension).")

# Select input method
upload_option = st.radio("Choose input method:", ("📤 Upload Image", "📷 Use Camera"))

image_data = None
image_filename = None

# File upload option
if upload_option == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_data = Image.open(uploaded_file).convert("RGB")
        image_filename = os.path.splitext(uploaded_file.name)[0]  # Remove extension

# Camera input option
elif upload_option == "📷 Use Camera":
    captured_image = st.camera_input("Take a photo using your webcam")
    if captured_image:
        image_data = Image.open(io.BytesIO(captured_image.getvalue())).convert("RGB")
        image_filename = "captured_image"  # No extension

# Display image and filename
if image_data:
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
    st.success(f"📝 **Uploaded Image Filename**: `{image_filename}`")