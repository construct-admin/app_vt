import os
import requests
import base64
import numpy as np
import cv2
import streamlit as st

def preprocess_image_for_gpt4(image_bytes, new_width=256, new_height=256, jpeg_quality=20):
    """
    1. Converts the uploaded image bytes to a NumPy array.
    2. Decodes it with OpenCV.
    3. Resizes it to (new_width x new_height) and compresses it as a JPEG with the specified quality.
    4. Returns the base64-encoded string of the compressed image.
    """
    # Convert image bytes to a NumPy array and decode
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    
    # Resize the image
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Compress as JPEG with aggressive quality settings
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    success, buffer = cv2.imencode('.jpg', resized, encode_params)
    if not success:
        raise ValueError("Failed to encode image")
    
    # Convert to base64 string
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

def analyze_image_gpt4_resized(image_bytes, prompt):
    """
    1. Resize & compress the image.
    2. Send the base64-encoded result to GPT-4 as part of a conversation.
    """
    # Set the system prompt
    system_prompt = "You accept images to generate description or alt text according to WCAG 2.2 AA accessibility standards."
    
    # Step 1: Resize & compress using the passed image_bytes
    try:
        image_base64 = preprocess_image_for_gpt4(image_bytes)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

    # Step 2: Prepare the conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "user", "content": (
            "Please analyze this compressed image data and provide me with a short visual transcript. "
            "Not more than 20 words:\n" + image_base64
        )}
    ]

    # Step 3: Call OpenAI API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4",
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(
            f"OpenAI API request failed with status {response.status_code}: {response.text}"
        )

    return response.json()
