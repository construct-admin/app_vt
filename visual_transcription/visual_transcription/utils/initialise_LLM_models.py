import streamlit as st
import os
import dotenv

# Load environment variables from a .env file if present
dotenv.load_dotenv()

# Define an OpenAI configuration dictionary.
# You can adjust the prompt, max_words, and other parameters as needed.
OpenAI_config = {
    "prompt": "Please analyze the visual content of the image. Ensure the description is concise and up to %MAX_WORDS% words.",
    "max_words": os.getenv("OPENAI_MAX_WORDS", "200"),  # default max words (can be overridden in a .env file)
    "selected_filters": [],
    "api_key": os.getenv("OPENAI_API_KEY")  # ensure your OpenAI API key is set in your environment
}

# Store the configuration in session_state for use in the app
st.session_state["gpt-4o"] = OpenAI_config

# Optionally, you can print or log to verify that your OpenAI configuration is loaded correctly.
st.write("OpenAI Configuration Loaded:", st.session_state["gpt-4o"])



""" from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
import streamlit as st
import os
import dotenv

dotenv.load_dotenv()

Azure_Vision_analyse_dict = {
    "region": "eastus",
    "client": ImageAnalysisClient(
        endpoint="https://rnd-calivision.cognitiveservices.azure.com/",
        credential=AzureKeyCredential(os.getenv("PERSONAL_AZURE_VISION_KEY"))
    ),
    "visual_features": None
}"""

