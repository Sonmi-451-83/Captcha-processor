import streamlit as st
import os
from pathlib import Path
import ultralytics
from ultralytics import YOLO
import cv2
import base64

st.set_page_config(page_title="Captcha Processor", page_icon="🔍", layout="centered")

# Background image
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


css = f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{get_base64_of_image("../streamlit_image.webp")}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Define the colors of the app
text_color = "white"
bg_color = "black"

# Apply custom CSS
st.markdown(f"""
    <style>        
        /* Changes the general text colour */
        html, body, [class*="st-"] {{
            color: {text_color};
        }}    
        /* Changes the colours in buttons */
        div.stButton > button:first-child {{
            background-color: {bg_color};
            color: {text_color};
        }}
        /* Changes the text colours in messages like success, warning, error */
        .stAlert {{
            color: {text_color};
            background-color: {bg_color};
        }}
        /* Changes the colours in text inputs */
        .stTextInput input {{
            color: {text_color};
            background-color: {bg_color};
        }}

    </style>
""", unsafe_allow_html=True)

# Loading our trained model
model_final = YOLO("./runs/detect/train2/weights/best.pt")

def real_chars(image_path):
    """
    Extracts the real characters from an image filename by matching it 
    with a reference list stored in a text file.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    list: The list of characters associated with the image filename.
    """
    files_all = "../data/raw/captchaobjectdetection/all_sequences.txt"
    chars_per_img = []

    with open(files_all, "r", encoding="utf-8") as fa:
        chars_per_img = [line.strip().split(",") for line in fa]
        filename = Path(image_path).stem
        chars = next((pair[1] for pair in chars_per_img if pair[0] == filename), [])
    
    return chars

def capcha_prediction(final_results, names):
    """
    Predicts the CAPTCHA text from object detection results by extracting and sorting character detections.

    Parameters:
    final_results (list): A list of detection results, where each result contains bounding boxes and class IDs.
    names (list): A list of class names corresponding to detected character indices.

    Returns:
    str: The predicted CAPTCHA text based on detected characters sorted from left to right.
    """
    detection_string = ""
    for result in final_results:
        boxes = result.boxes.xyxy
        class_ids = result.boxes.cls

        detections = sorted(zip(boxes, class_ids), key=lambda x: x[0][0])

        for box, class_id in detections:
            detection_string += f"{names[int(class_id)]}"

    return detection_string


def captcha_boxes_prediction(final_results, image_path):
    """
    Draws bounding boxes around detected CAPTCHA characters in an image.

    Parameters:
    final_results (list): A list of detection results containing bounding boxes.
    image_path (str): Path to the image file.

    Returns:
    numpy.ndarray: The image with drawn bounding boxes.
    """
    image = cv2.imread(image_path)
    image_hight = image.shape[0]
    image_width = image.shape[1]

    for result in final_results:
        boxes = result.boxes.xyxy

        for box in boxes:
            x0, y0, x1, y1 = map(int, box) 
            img = cv2.rectangle(image, (x0, y0), (x1, y1), (80, 80, 180), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


st.title("🔍 CAPTCHA Processor")
st.markdown("### 🚀 Extract text from images using a YOLO-based model")

image_path = st.text_input("📸 **Enter the image path:**", placeholder="e.g., /path/to/image.png")

if st.button("🔎 Process Image"):
    if image_path:
        try:
            with st.spinner("Processing... ⏳"):
                final_results = model_final(image_path)  # Executes the model
                result = capcha_prediction(final_results, model_final.names)  # Predicts the characters
                chars = real_chars(image_path)  # Obtains real characters from file
                img = captcha_boxes_prediction(final_results, image_path)  # Boxes predictions from models

            st.divider()
            st.subheader("📷 Uploaded Image")
            st.image(image_path, caption="Original Image", use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("✅ **Processed Text:**")
                st.success(result)

                st.subheader("🔠 **Real Text:**")
                st.info(chars)

            with col2:
                st.subheader("📦 **YOLO Predictions:**")
                st.image(img, caption="Detected Characters", use_container_width=True)

            if (chars == result):
                st.success("✔️ Process completed successfully!")
            else:
                st.succes("⚠️ Ups, not a perfect prediction, nobody is perfect!")

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

    else:
        st.warning("⚠️ Please enter a valid image path.")
