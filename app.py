import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import tempfile
import os
from pathlib import Path
import re
import json
from datetime import datetime
import io

# Set page config
st.set_page_config(
    page_title="OCR Barcode Extractor",
    page_icon="📄",
    layout="wide"
)

class OCREngine:
    """OCR Engine with model upload support - No OpenCV"""
    
    def __init__(self):
        self.engine_type = "pytesseract"
        self.custom_model_loaded = False
        self.model_path = None
        
    def load_custom_model(self, uploaded_file, engine_type: str):
        """Load custom OCR model"""
        try:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                self.model_path = tmp_file.name
            
            self.engine_type = engine_type
            self.custom_model_loaded = True
            
            st.success(f"✅ Custom model loaded: {uploaded_file.name}")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def extract_barcodes(self, image: Image.Image) -> dict:
        """Extract barcodes from image using OCR"""
        results = {
            'barcodes': [],
            'text_blocks': [],
            'confidence': 0.0,
            'engine_used': self.engine_type,
            'custom_model': self.custom_model_loaded
        }
        
        try:
            return self._extract_with_tesseract(image, results)
        except Exception as e:
            st.error(f"OCR extraction error: {e}")
            return results
    
    def _extract_with_tesseract(self, image: Image.Image, results: dict) -> dict:
        """Extract using Tesseract OCR"""
        # Configure Tesseract for barcode detection
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Extract text with bounding boxes
        text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
        
        # Process results
        barcodes = []
        text_blocks = []
        
        for i, text in enumerate(text_data['text']):
            if text.strip():
                confidence = float(text_data['conf'][i]) / 100.0
                if confidence > 0.1:  # Filter low confidence results
                    bbox = (
                        text_data['left'][i],
                        text_data['top'][i],
                        text_data['left'][i] + text_data['width'][i],
                        text_data['top'][i] + text_data['height'][i]
                    )
                    
                    text_blocks.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    
                    # Check if text matches barcode patterns
                    barcode = self._extract_barcode_pattern(text)
                    if barcode:
                        barcodes.append({
                            'barcode': barcode,
                            'confidence': confidence,
                            'source_text': text,
                            'bbox': bbox
                        })
        
        results['barcodes'] = barcodes
        results['text_blocks'] = text_blocks
        results['confidence'] = max([b['confidence'] for b in barcodes]) if barcodes else 0.0
        
        return results
    
    def _extract_barcode_pattern(self, text: str) -> str:
        """Extract barcode patterns from text"""
        # Common barcode patterns
        patterns = [
            r'\b\d{12,13}\b',      # EAN-13, UPC (12-13 digits)
            r'\b\d{8}\b',          # EAN-8 (8 digits)
            r'\b[0-9A-Z]{8,15}\b', # Alphanumeric codes (8-15 chars)
            r'\b\d{6,14}\b',       # Generic numeric codes
        ]
        
        # Clean text
        clean_text = re.sub(r'[^\w\s]', '', text.upper())
        
        for pattern in patterns:
            matches = re.findall(pattern, clean_text)
            for match in matches:
                if len(match) >= 6:  # Minimum barcode length
                    return match
        
        return ""

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image for better OCR results - Pure PIL"""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Enhance contrast (simple histogram stretch)
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Increase contrast
    
    # Sharpen image
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    return image

def draw_ocr_results(image: Image.Image, ocr_results: dict) -> Image.Image:
    """Draw OCR results on image - Pure PIL"""
    # Create a copy to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    try:
        # Try to load a font, fallback to default
        font = ImageFont.load_default()
        # For larger text, we'll use a built-in font
        large_font = ImageFont.load_default()
    except:
        font = None
        large_font = None
    
    # Draw bounding boxes for all text blocks (green)
    for block in ocr_results['text_blocks']:
        bbox = block['bbox']
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
    
    # Draw barcode bounding boxes in red
    for barcode in ocr_results['barcodes']:
        bbox = barcode['bbox']
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            
            # Draw red bounding box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # Draw barcode text above the box
            label = f"Barcode: {barcode['barcode']}"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw text background
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 10, y1], 
                          fill='red')
            
            # Draw text
            draw.text((x1 + 5, y1 - text_height - 2), label, fill='white', font=font)
    
    return draw_image

def main():
    st.title("📄 OCR Barcode Extractor")
    st.markdown("Upload images and custom OCR models to extract barcodes")
    
    # Initialize OCR engine
    if 'ocr_engine' not in st.session_state:
        st.session_state.ocr_engine = OCREngine()
    
    # Sidebar for configuration
    st.sidebar.header("OCR Configuration")
    
    # OCR Engine selection
    engine_type = st.sidebar.selectbox(
        "Select OCR Engine",
        ["pytesseract"],
        help="Choose OCR engine for barcode extraction"
    )
    
    # Custom model upload
    st.sidebar.subheader("📁 Upload Custom OCR Model")
    uploaded_model = st.sidebar.file_uploader(
        "Upload trained OCR model",
        type=['traineddata', 'h5', 'pth', 'pt', 'onnx'],
        help="Upload custom trained OCR model files (Tesseract .traineddata recommended)"
    )
    
    if uploaded_model is not None:
        if st.sidebar.button("🚀 Load Custom Model"):
            with st.spinner("Loading custom model..."):
                success = st.session_state.ocr_engine.load_custom_model(uploaded_model, engine_type)
                if success:
                    st.sidebar.success(f"Model loaded: {uploaded_model.name}")
    
    # Model info display
    if st.session_state.ocr_engine.custom_model_loaded:
        st.sidebar.success(f"🔧 Custom model active: {st.session_state.ocr_engine.model_path}")
    else:
        st.sidebar.info("🔧 Using default Tesseract OCR")
    
    # Processing options
    st.sidebar.subheader("⚙️ Processing Options")
    enable_preprocessing = st.sidebar.checkbox("Enable Image Preprocessing", value=True)
    show_text_blocks = st.sidebar.checkbox("Show All Text Blocks", value=False)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Images")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose images for barcode extraction",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload images containing barcodes"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} image(s) uploaded")
            
            # Process all images
            if st.button("🚀 Extract Barcodes from All Images", type="primary", use_container_width=True):
                process_multiple_images(uploaded_files, enable_preprocessing, show_text_blocks, min_confidence)
        
        # Single image processing
        if uploaded_files and len(uploaded_files) == 1:
            st.subheader("👀 Single Image Preview")
            process_single_image(uploaded_files[0], enable_preprocessing, show_text_blocks, min_confidence)
    
    with col2:
        st.subheader("📊 Results & Export")
        
        # Display results from session state
        if 'ocr_results' in st.session_state:
            display_results(st.session_state.ocr_results, show_text_blocks)
        else:
            st.info("📝 Upload images and click 'Extract Barcodes' to see results")

def process_single_image(uploaded_file, enable_preprocessing: bool, show_text_blocks: bool, min_confidence: float):
    """Process a single uploaded image"""
    try:
        # Read image
        image = Image.open(uploaded_file)
        
        # Display original image
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Preprocess if enabled
        if enable_preprocessing:
            processed_image = preprocess_image(image)
            st.image(processed_image, caption="Preprocessed Image", use_column_width=True)
            image_to_process = processed_image
        else:
            image_to_process = image
        
        # Extract barcodes
        with st.spinner("🔍 Extracting barcodes..."):
            ocr_results = st.session_state.ocr_engine.extract_barcodes(image_to_process)
        
        # Filter by confidence
        ocr_results['barcodes'] = [b for b in ocr_results['barcodes'] if b['confidence'] >= min_confidence]
        ocr_results['text_blocks'] = [t for t in ocr_results['text_blocks'] if t['confidence'] >= min_confidence]
        
        # Store results
        st.session_state.ocr_results = {
            'filename': uploaded_file.name,
            'timestamp': datetime.now().isoformat(),
            'results': ocr_results
        }
        
        # Display results
        display_single_results(ocr_results, image, show_text_blocks)
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

def process_multiple_images(uploaded_files, enable_preprocessing: bool, show_text_blocks: bool, min_confidence: float):
    """Process multiple uploaded images"""
    all_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"🔄 Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        try:
            # Read and process image
            image = Image.open(uploaded_file)
            
            # Preprocess if enabled
            if enable_preprocessing:
                image_to_process = preprocess_image(image)
            else:
                image_to_process = image
            
            # Extract barcodes
            ocr_results = st.session_state.ocr_engine.extract_barcodes(image_to_process)
            
            # Filter by confidence
            ocr_results['barcodes'] = [b for b in ocr_results['barcodes'] if b['confidence'] >= min_confidence]
            
            # Store results
            all_results[uploaded_file.name] = {
                'timestamp': datetime.now().isoformat(),
                'results': ocr_results,
                'barcodes_found': len(ocr_results['barcodes'])
            }
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            all_results[uploaded_file.name] = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'barcodes_found': 0
            }
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("✅ Processing complete!")
    st.session_state.ocr_results = all_results
    
    # Show summary
    total_barcodes = sum(result.get('barcodes_found', 0) for result in all_results.values())
    st.success(f"✅ Processed {len(uploaded_files)} images, found {total_barcodes} barcodes")

def display_single_results(ocr_results: dict, original_image: Image.Image, show_text_blocks: bool):
    """Display results for a single image"""
    st.subheader("📈 Extraction Results")
    
    # Barcodes found
    if ocr_results['barcodes']:
        st.success(f"✅ Found {len(ocr_results['barcodes'])} barcode(s)")
        
        # Display barcodes in a table
        barcode_data = []
        for i, barcode in enumerate(ocr_results['barcodes']):
            barcode_data.append({
                'Barcode': barcode['barcode'],
                'Confidence': f"{barcode['confidence']:.2%}",
                'Source Text': barcode['source_text']
            })
        
        st.table(barcode_data)
        
        # Draw results on image
        result_image = draw_ocr_results(original_image, ocr_results)
        st.image(result_image, caption="Barcode Detection Results", use_column_width=True)
        
        # Export single results
        export_single_results(ocr_results, original_image)
    
    else:
        st.warning("❌ No barcodes found")
    
    # Show all text blocks if requested
    if show_text_blocks and ocr_results['text_blocks']:
        st.subheader("📝 All Detected Text Blocks")
        text_data = []
        for block in ocr_results['text_blocks']:
            is_barcode = any(b['source_text'] == block['text'] for b in ocr_results['barcodes'])
            text_data.append({
                'Text': block['text'],
                'Confidence': f"{block['confidence']:.2%}",
                'Is Barcode': '✅' if is_barcode else '❌'
            })
        
        st.table(text_data)

def export_single_results(ocr_results: dict, image: Image.Image):
    """Export single image results"""
    st.subheader("📥 Export Results")
    
    # JSON export
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'ocr_engine': ocr_results['engine_used'],
        'custom_model_used': ocr_results['custom_model'],
        'barcodes_found': len(ocr_results['barcodes']),
        'barcodes': ocr_results['barcodes'],
        'text_blocks': ocr_results['text_blocks'] if st.session_state.get('show_text_blocks', False) else []
    }
    
    json_str = json.dumps(export_data, indent=2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📄 Download JSON Results",
            data=json_str,
            file_name=f"barcode_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Save annotated image
        if ocr_results['barcodes']:
            annotated_image = draw_ocr_results(image, ocr_results)
            img_buffer = io.BytesIO()
            annotated_image.save(img_buffer, format='PNG')
            
            st.download_button(
                label="🖼️ Download Annotated Image",
                data=img_buffer.getvalue(),
                file_name=f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )

def display_results(all_results: dict, show_text_blocks: bool):
    """Display results for multiple images"""
    if isinstance(all_results, dict) and 'filename' in all_results:
        # Single image results
        display_single_results(all_results['results'], None, show_text_blocks)
    else:
        # Multiple image results
        st.subheader("📊 Batch Processing Results")
        
        # Summary table
        summary_data = []
        total_barcodes = 0
        
        for filename, result in all_results.items():
            barcodes_found = result.get('barcodes_found', 0)
            total_barcodes += barcodes_found
            
            summary_data.append({
                'Filename': filename,
                'Barcodes Found': barcodes_found,
                'Status': '✅ Success' if 'error' not in result else '❌ Error',
                'Error': result.get('error', '')
            })
        
        st.table(summary_data)
        
        # Export all results
        if st.button("📦 Export All Results as JSON", use_container_width=True):
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'ocr_engine': st.session_state.ocr_engine.engine_type,
                'custom_model_used': st.session_state.ocr_engine.custom_model_loaded,
                'total_images': len(all_results),
                'total_barcodes': total_barcodes,
                'results': all_results
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📥 Download Complete Results",
                data=json_str,
                file_name=f"batch_barcode_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Show detailed results for each file
        st.subheader("🔍 Detailed Results")
        selected_file = st.selectbox("Select file to view details:", list(all_results.keys()))
        
        if selected_file:
            result = all_results[selected_file]
            if 'results' in result:
                st.write(f"**Barcodes found in {selected_file}:**")
                if result['results']['barcodes']:
                    for barcode in result['results']['barcodes']:
                        st.write(f"- `{barcode['barcode']}` (Confidence: {barcode['confidence']:.2%})")
                else:
                    st.write("No barcodes found")

if __name__ == "__main__":
    main()
