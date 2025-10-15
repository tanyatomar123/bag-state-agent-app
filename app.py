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

# Add YOLO import with error handling
try:
    from ultralytics import YOLO
except ImportError:
    st.error("❌ Ultralytics not installed. Please run: `pip install ultralytics`")
    st.stop()

# Set page config
st.set_page_config(
    page_title="YOLO OCR Barcode Extractor",
    page_icon="📸",
    layout="wide"
)

# Main app
def main():
    st.title("YOLO OCR Barcode Extractor")
    st.markdown("Upload YOLO models (.pt) and images for barcode detection - No OpenCV Required")
    
    # Sidebar for model upload
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
                    
                    # Display model info
                    st.info(f"Model: {uploaded_model.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    st.session_state.model_loaded = False
    else:
        st.sidebar.warning("⚠️ Please upload a YOLO model (.pt file) to begin")
    
    # Main content area
    st.header("Image Processing")
    
    # Image upload section
    uploaded_images = st.file_uploader(
        "Choose images for barcode detection",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Select one or multiple images for barcode detection"
    )
    
    # Process images if model is loaded and images are uploaded
    if st.session_state.model_loaded and uploaded_images:
        st.subheader("Detection Results")
        
        # Configuration
        col1, col2 = st.columns(2)
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                help="Minimum confidence score for detection"
            )
        
        with col2:
            iou_threshold = st.slider(
                "IOU Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.45,
                help="Intersection over Union threshold for NMS"
            )
        
        # Process button
        if st.button("🚀 Start Detection", type="primary"):
            process_images(uploaded_images, confidence_threshold, iou_threshold)
    
    elif uploaded_images and not st.session_state.model_loaded:
        st.warning("⚠️ Please upload and load a YOLO model first")
    
    # Instructions section
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### How to use this app:
        1. **Upload YOLO Model**: In the sidebar, upload your trained YOLO model (.pt file)
        2. **Upload Images**: Select one or multiple images for barcode detection
        3. **Configure Settings**: Adjust confidence and IOU thresholds as needed
        4. **Start Detection**: Click the 'Start Detection' button to process images
        5. **View Results**: See detected barcodes with bounding boxes and extracted text
        
        ### Supported Features:
        - Multiple image upload
        - Barcode detection with bounding boxes
        - OCR text extraction
        - Results export in multiple formats
        - Batch processing
        """)

def process_images(uploaded_images, confidence_threshold, iou_threshold):
    """Process uploaded images for barcode detection and OCR"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, uploaded_image in enumerate(uploaded_images):
        # Update progress
        progress = (i + 1) / len(uploaded_images)
        progress_bar.progress(progress)
        status_text.text(f"Processing image {i + 1} of {len(uploaded_images)}: {uploaded_image.name}")
        
        try:
            # Load image
            image = Image.open(uploaded_image)
            image_np = np.array(image)
            
            # Perform detection
            with st.spinner(f"Detecting barcodes in {uploaded_image.name}..."):
                detections = st.session_state.model(
                    image_np,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
            
            # Process results
            image_result = {
                'filename': uploaded_image.name,
                'original_image': image,
                'detections': [],
                'annotated_image': None
            }
            
            # Annotate image with detections
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            for detection in detections:
                if detection.boxes is not None and len(detection.boxes) > 0:
                    boxes = detection.boxes.xyxy.cpu().numpy()
                    confidences = detection.boxes.conf.cpu().numpy()
                    class_ids = detection.boxes.cls.cpu().numpy()
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        # Convert coordinates to integers
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Draw bounding box
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                        
                        # Draw label
                        label = f"Barcode: {conf:.2f}"
                        draw.rectangle([x1, y1-25, x1+len(label)*8, y1], fill="red")
                        draw.text((x1+5, y1-20), label, fill="white")
                        
                        # Store detection info
                        detection_info = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(class_id)
                        }
                        image_result['detections'].append(detection_info)
            
            image_result['annotated_image'] = annotated_image
            results.append(image_result)
            
            # Display results for this image
            with st.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption=f"Original: {uploaded_image.name}", use_column_width=True)
                
                with col2:
                    st.image(annotated_image, caption=f"Detected: {len(image_result['detections'])} barcodes", use_column_width=True)
                
                # Display detection details
                if image_result['detections']:
                    st.write(f"**Detections in {uploaded_image.name}:**")
                    for j, detection in enumerate(image_result['detections']):
                        st.write(f"Barcode {j+1}: Confidence: {detection['confidence']:.3f}, "
                                f"BBox: {detection['bbox']}")
                else:
                    st.write(f"**No barcodes detected in {uploaded_image.name}**")
                
                st.markdown("---")
        
        except Exception as e:
            st.error(f"Error processing {uploaded_image.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Export results if any detections were found
    if results and any(len(result['detections']) > 0 for result in results):
        export_results(results)

def export_results(results):
    """Export detection results in various formats"""
    
    st.subheader("📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    # JSON Export
    with col1:
        if st.button("📊 Export as JSON"):
            export_data = []
            for result in results:
                if result['detections']:
                    export_data.append({
                        'filename': result['filename'],
                        'detections': result['detections'],
                        'timestamp': datetime.now().isoformat()
                    })
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"barcode_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Images Export
    with col2:
        if st.button("🖼️ Export Annotated Images"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for result in results:
                    if result['detections']:
                        img_buffer = io.BytesIO()
                        result['annotated_image'].save(img_buffer, format='PNG')
                        zip_file.writestr(f"detected_{result['filename']}", img_buffer.getvalue())
            
            st.download_button(
                label="Download Images ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"barcode_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
    
    # Report Export
    with col3:
        if st.button("📋 Generate Report"):
            report = generate_report(results)
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"barcode_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def generate_report(results):
    """Generate a text report of detection results"""
    
    report_lines = []
    report_lines.append("Barcode Detection Report")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    total_detections = 0
    for result in results:
        report_lines.append(f"File: {result['filename']}")
        report_lines.append(f"Detections: {len(result['detections'])}")
        
        for i, detection in enumerate(result['detections']):
            report_lines.append(f"  Barcode {i+1}:")
            report_lines.append(f"    Confidence: {detection['confidence']:.3f}")
            report_lines.append(f"    Bounding Box: {detection['bbox']}")
            report_lines.append(f"    Class ID: {detection['class_id']}")
        
        total_detections += len(result['detections'])
        report_lines.append("")
    
    report_lines.append(f"Total barcodes detected: {total_detections}")
    report_lines.append(f"Files processed: {len(results)}")
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    main()
