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
    1. Resize & compress the image from image_bytes.
    2. Send the base64-encoded result to GPT-4 as part of a conversation.
    """
    # Set the system prompt
    system_prompt = "You accept images to generate description or alt text according to WCAG 2.2 AA accessibility standards."
    # Step 1: Resize & compress using the passed image bytes
    try:
        image_base64 = preprocess_image_for_gpt4(image_bytes)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

    # Step 2: Prepare the conversation (system prompt, user instructions, and image data)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "user", "content": (
            "Please analyze this compressed image data and provide a short visual transcript in no more than 20 words:\n" +
            image_base64
        )}
    ]

    # Step 3: Call OpenAI API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY environment variable not set.")
        return None

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
        st.error(f"OpenAI API request failed with status {response.status_code}: {response.text}")
        return None
    return response.json()

############################################
# The following is an example integration in Streamlit:
############################################

# Initialize session state values
if "saved_frames" not in st.session_state:
    st.session_state.saved_frames = dict()
if 'video' not in st.session_state:
    st.session_state.video = None
if 'frame_number' not in st.session_state:
    st.session_state.frame_number = 0
if 'total_frames' not in st.session_state:
    st.session_state.total_frames = 0
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False

st.title('Video Transcription Service')

# File Uploader: Drag and Drop a Video File
if not st.session_state.uploaded:
    uploaded_file = st.file_uploader('Drag and drop a video file here', type=['mp4', 'avi', 'mov'])
else:
    uploaded_file = None

if uploaded_file is not None:
    if not st.session_state.uploaded:
        st.success('Video uploaded successfully! Processing transcription...')
        # Save the uploaded video to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.write(f'Temporary file path: {temp_file_path}')
        if not os.path.exists(temp_file_path):
            st.error('Temporary file was not created successfully.')
        else:
            st.session_state.video = cv2.VideoCapture(temp_file_path)
            if st.session_state.video.isOpened():
                st.session_state.total_frames = int(st.session_state.video.get(cv2.CAP_PROP_FRAME_COUNT))
                st.session_state.uploaded = True
            else:
                st.error('Could not open video file.')

# Display Video Frames and Controls
if st.session_state.uploaded:
    frame_number = st.slider('Select frame', 0, st.session_state.total_frames - 1, st.session_state.frame_number)
    st.session_state.frame_number = frame_number
    st.write(f"Frame number: {st.session_state.frame_number}")

    stframe = st.empty()
    st.session_state.video.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_number)
    ret, frame = st.session_state.video.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels='RGB')
    else:
        st.error('Could not read the frame.')

# Navigation and Save Frame buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Move Left'):
        if st.session_state.frame_number > 0:
            st.session_state.frame_number -= 1
with col2:
    if st.button('Save Frame Index'):
        st.session_state.video.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_number)
        ret, frame = st.session_state.video.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.saved_frames[st.session_state.frame_number] = {
                'frame': frame_rgb,
                'has_visual_transcripts': False,
                'visual_transcripts': None
            }
            st.write(f'Saved frame index: {st.session_state.frame_number}')
        else:
            st.error('Could not capture the frame to save.')
with col3:
    if st.button('Move Right'):
        if st.session_state.frame_number < st.session_state.total_frames - 1:
            st.session_state.frame_number += 1

# Sidebar: Display Saved Frames as Clickable Cards
with st.sidebar:
    st.markdown("### Selected Frames")
    for frame_index, frame_info in sorted(st.session_state.saved_frames.items()):
        # Convert frame to base64 for display
        from io import BytesIO
        from PIL import Image
        buffered = BytesIO()
        pil_image = Image.fromarray(frame_info['frame'])
        pil_image.save(buffered, format="JPEG")
        base64_img = base64.b64encode(buffered.getvalue()).decode()
        
        transcript_text = frame_info['visual_transcripts'] if frame_info['visual_transcripts'] else 'No transcript yet'
        
        st.markdown(f"""
            <div style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;">
                <h4>Frame {frame_index}</h4>
                <img src="data:image/jpeg;base64,{base64_img}" style="width:100%;" />
                <p>{transcript_text}</p>
            </div>
        """, unsafe_allow_html=True)
        
        if not frame_info['has_visual_transcripts']:
            if st.button(f"Transcribe frame {frame_index}", key=f"btn_{frame_index}"):
                # Call our function with the saved frame converted back to bytes
                # First, encode the frame as JPEG bytes
                ret2, buf = cv2.imencode('.jpg', frame_info['frame'])
                if ret2:
                    image_bytes = buf.tobytes()
                    response = analyze_image_gpt4_resized(image_bytes, "Generate alt text for the image.")
                    if response:
                        choices = response.get("choices", [])
                        if choices:
                            message = choices[0]["message"]["content"]
                            st.session_state.saved_frames[frame_index]['visual_transcripts'] = message
                            st.session_state.saved_frames[frame_index]['has_visual_transcripts'] = True
                            st.success(f"Visual transcription updated for frame {frame_index}.")
                else:
                    st.error("Could not encode frame for transcription.")

if st.session_state.saved_frames:
    st.sidebar.markdown("### Saved Transcript Messages")
    for frame_index, frame_info in sorted(st.session_state.saved_frames.items()):
        if frame_info.get("visual_transcripts"):
            st.sidebar.write(f"Frame {frame_index}: {frame_info['visual_transcripts']}")
