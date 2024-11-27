import os
import base64
from io import BytesIO 
from mistralai import Mistral
from PIL import Image
import streamlit as st

# Function to encode image to base64
def encode_image(image):
    """Encode the image to base64."""
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        return f"Error encoding image: {e}"

# Function to analyze image with Mistral
def analyze_image_with_mistral(image_data, text_prompt):
    """
    Analyze a medical image with the Mistral API.
    """
    # Set up API key and model
    api_key = "KEY"  
    model = "pixtral-12b-2409"
    
    if not api_key:
        return "Error: MISTRAL_API_KEY not set in environment."
    
    # Initialize the Mistral client
    client = Mistral(api_key=api_key)

    # Define the messages for the chat
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"}
            ]
        }
    ]

    try:
        # Get the chat response
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with Mistral API: {e}"

# Streamlit UI
st.title("Medical Image Analysis with Mistral")
st.write("Upload a medical image and provide a prompt for analysis.")

# File uploader
uploaded_image = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Encode image to base64
    encoded_image = encode_image(image)
    if "Error" in encoded_image:
        st.error(encoded_image)
    else:
        # Input prompt
        text_prompt = st.text_input("Enter a prompt for analysis:", "Describe abnormalities in this image.")
        
        if st.button("Analyze with Mistral"):
            st.write("Analyzing...")
            result = analyze_image_with_mistral(encoded_image, text_prompt)
            st.subheader("Mistral Analysis Result")
            st.write(result)
