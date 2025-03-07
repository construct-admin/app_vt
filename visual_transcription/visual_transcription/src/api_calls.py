import os
import requests
import base64
import numpy as np
import cv2

def preprocess_image_for_gpt4(frame, new_width=256, new_height=256, jpeg_quality=20):
    """
    Resizes the frame to (new_width x new_height),
    compresses it as a JPEG with the specified jpeg_quality (0-100),
    and returns the base64-encoded string of the compressed image.
    """
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

def analyze_image_gpt4_resized(frame, prompt):
    """
    Resizes & compresses the given image (frame), converts it to a base64 string,
    and sends it to GPT-4 via the ChatCompletion API along with a conversation prompt.
    
    Returns the JSON response from the API.
    """
    # Set the system prompt to instruct GPT-4 on how to process the image
    system_prompt = "You accept images to generate description or alt text according to WCAG 2.2 AA accessibility standards."
    
    try:
        image_base64 = preprocess_image_for_gpt4(frame, new_width=256, new_height=256, jpeg_quality=20)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")
    
    # Build the conversation messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "user", "content": (
            "Please analyze this compressed image and provide me with a short visual transcript in no more than 20 words:\n" +
            image_base64
        )}
    ]
    
    # Retrieve API key from environment variables
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
        "supports_image": True,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"OpenAI API request failed with status {response.status_code}: {response.text}")
    
    return response.json()
