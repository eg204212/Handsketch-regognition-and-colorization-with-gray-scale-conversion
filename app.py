import streamlit as st
from streamlit_drawable_canvas import st_canvas
from src.grayscale import convert_to_grayscale
from src.colorize import colorize_any
from src.recognition import predict_sketch
import numpy as np
import cv2
import os
from datetime import datetime

st.title("Handsketch Recognition and Colorization")

# Create datasets folder if not exists
os.makedirs("datasets", exist_ok=True)

# --- User choice: upload or draw ---
option = st.radio("Choose input method:", ["Upload Sketch", "Draw Sketch"])

input_image = None

# --- Upload Sketch ---
if option == "Upload Sketch":
    uploaded_file = st.file_uploader("Upload a sketch image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img_path = f"datasets/{uploaded_file.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_image = img_path

# --- Draw Sketch ---
elif option == "Draw Sketch":
    # Button to reset canvas
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = f"canvas_{datetime.now().timestamp()}"

    if st.button("🗑️ New Sketch"):
        st.session_state.canvas_key = f"canvas_{datetime.now().timestamp()}"

    # Create canvas
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=5,
        stroke_color="black",
        background_color="white",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key
    )

    # Save drawing if available
    if canvas_result.image_data is not None:
        drawn_img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2RGB)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_image = f"datasets/drawn_sketch_{timestamp}.png"
        cv2.imwrite(input_image, drawn_img)

# --- Process the image if available ---
if input_image:
    # Convert to grayscale
    gray_path = convert_to_grayscale(input_image)

    # Predict label
    label, confidence = predict_sketch(gray_path)
    prediction_text = f"Predicted Label: **{label}** (Confidence: {confidence*100:.2f}%)"

    # Colorize sketch
    colorized_path = colorize_any(gray_path)

    # --- Display images in horizontal line ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(input_image, caption="Original Sketch", use_container_width=True)
        st.markdown(prediction_text)
    with col2:
        st.image(gray_path, caption="Grayscale Sketch", use_container_width=True)
    with col3:
        st.image(colorized_path, caption="Colorized Sketch", use_container_width=True)
