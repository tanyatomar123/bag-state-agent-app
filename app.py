import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import torch
import tempfile
import os
from pathlib import Path
import re
import json
from datetime import datetime
import io
import zipfile
import subprocess
import sys

# Enhanced YOLO import with installation guidance
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="YOLO OCR Barcode Extractor",
    page_icon="📸",
    layout="wide"
)

def main():
    st.title("YOLO OCR Barcode Extractor")
    st.markdown("Upload YOLO models (.pt) and images for barcode detection - No OpenCV Required")
    
    # Show installation instructions if ultralytics is not available
    if not YOLO_AVAILABLE:
        st.error("❌ Ultralytics not installed")
        
        st.markdown("""
        ### To fix this issue, please run one of the following commands:
        
        **Option 1: Install ultralytics only**
        ```bash
        pip install ultralytics
        ```
        
        **Option 2: Install all required packages**
        ```bash
        pip install ultralytics streamlit torch pillow numpy
        ```
        
        **Option 3: If you have a requirements.txt file**
        ```bash
        pip install -r requirements.txt
        ```
        
        **After installation, restart this app.**
        """)
        
        # Optional: Auto-install button (use with caution)
        if st.button("🔄 Try Auto-install ultralytics"):
            with st.spinner("Installing ultralytics..."):
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                    st.success("✅ Installation successful! Please restart the app.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Installation failed: {str(e)}")
        
        return  # Stop execution if ultralytics is not available
    
    # If ultralytics is available, continue with the app
    st.success("✅ Ultralytics is available!")
    
    # Rest of your app code here...
    st.sidebar.header("YOLO Model Configuration")
    
    # Model upload section
    uploaded_model = st.sidebar.file_uploader(
        "Upload YOLO Model (.pt file)",
        type=['pt'],
        help="Upload your trained YOLO model weights"
    )
    
    # Initialize session state for model
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    # Load model if uploaded
    if uploaded_model is not None:
        with st.sidebar:
            with st.spinner("Loading YOLO model..."):
                try:
                    # Save uploaded model to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                        tmp_file.write(uploaded_model.read())
                        model_path = tmp_file.name
                    
                    # Load YOLO model
                    st.session_state.model = YOLO(model_path)
                    st.session_state.model_loaded = True
                    
                    st.success("✅ YOLO model loaded successfully!")
                    st.info(f"Model: {uploaded_model.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    st.session_state.model_loaded = False
    else:
        st.sidebar.warning("⚠️ Please upload a YOLO model (.pt file) to begin")
    
    # Continue with the rest of your app logic...

if __name__ == "__main__":
    main()
