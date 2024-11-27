import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import streamlit as st

# Initialize the OpenAI client
client = OpenAI(
    api_key="KEY"
)


def preprocess_image(image):
    """Convert an image to a base64-encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64_str}"

def analyze_image_with_openai(image_data, text_prompt):
    """Analyze a medical image using OpenAI's chat API."""
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            }
        ],
    )
    return response["choices"][0]["message"]["content"]



# Streamlit App
st.title("Medical Image Analysis with OpenAI GPT-4o")
st.write("Upload a medical image and provide a prompt for analysis.")

# File uploader for medical images
uploaded_image = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image_data = preprocess_image(image)
    
    # Input prompt
    text_prompt = st.text_input("Enter a prompt for analysis:", "Describe abnormalities in this image.")
    
    if st.button("Analyze"):
        try:
            st.write("Analyzing with GPT-4o")
            result = analyze_image_with_openai(image_data, text_prompt)
            st.subheader("GPT-4o Analysis Result")
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
