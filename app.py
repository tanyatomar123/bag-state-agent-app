import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import tempfile
import os
from pathlib import Path
import re
import json
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="OCR Barcode Extractor",
    page_icon="📄",
    layout="wide"
)

class OCREngine:
    """OCR Engine with multiple backends and model upload support"""
    
    def __init__(self):
        self.engine_type = "pytesseract"
        self.custom_model_loaded = False
        self.model_path = None
        self.engine = None
        
    def load_custom_model(self, uploaded_file, engine_type: str):
        """Load custom OCR model"""
        try:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                self.model_path = tmp_file.name
            
            self.engine_type = engine_type
            self.custom_model_loaded = True
            
            # Initialize engine based on type
            if engine_type == "pytesseract":
                # For Tesseract, we just note the custom model path
                st.success(f"✅ Custom Tesseract model loaded: {uploaded_file.name}")
                
            elif engine_type == "paddleocr":
                try:
                    from paddleocr import PaddleOCR
                    # Load custom PaddleOCR model
                    self.engine = PaddleOCR(
                        det_model_dir=self.model_path if 'det' in uploaded_file.name else None,
                        rec_model_dir=self.model_path if 'rec' in uploaded_file.name else None,
                        cls_model_dir=self.model_path if 'cls' in uploaded_file.name else None,
                        use_angle_cls=True,
                        lang='en',
                        show_log=False
                    )
                    st.success(f"✅ Custom PaddleOCR model loaded: {uploaded_file.name}")
                except ImportError:
                    st.warning("PaddleOCR not available, using pytesseract")
                    self.engine_type = "pytesseract"
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def extract_barcodes(self, image: np.ndarray) -> dict:
        """Extract barcodes from image using selected OCR engine"""
        results = {
            'barcodes': [],
            'text_blocks': [],
            'confidence': 0.0,
            'engine_used': self.engine_type,
            'custom_model': self.custom_model_loaded
        }
        
        try:
            if self.engine_type == "pytesseract":
                return self._extract_with_tesseract(image, results)
            elif self.engine_type == "paddleocr" and self.engine:
                return self._extract_with_paddleocr(image, results)
            else:
                return self._extract_with_tesseract(image, results)
                
        except Exception as e:
            st.error(f"OCR extraction error: {e}")
            return results
    
    def _extract_with_tesseract(self, image: np.ndarray, results: dict) -> dict:
        """Extract using Tesseract OCR"""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Configure Tesseract for barcode detection
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Extract text
        text_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=custom_config)
        
        # Process results
        barcodes = []
        text_blocks = []
        
        for i, text in enumerate(text_data['text']):
            if text.strip():
                confidence = float(text_data['conf'][i]) / 100.0
                if confidence > 0.1:  # Filter low confidence results
                    text_blocks.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': (
                            text_data['left'][i],
                            text_data['top'][i],
                            text_data['left'][i] + text_data['width'][i],
                            text_data['top'][i] + text_data['height'][i]
                        )
                    })
                    
                    # Check if text matches barcode patterns
                    barcode = self._extract_barcode_pattern(text)
                    if barcode:
                        barcodes.append({
                            'barcode': barcode,
                            'confidence': confidence,
                            'source_text': text,
                            'bbox': (
                                text_data['left'][i],
                                text_data['top'][i],
                                text_data['left'][i] + text_data['width'][i],
                                text_data['top'][i] + text_data['height'][i]
                            )
                        })
        
        results['barcodes'] = barcodes
        results['text_blocks'] = text_blocks
        results['confidence'] = max([b['confidence'] for b in barcodes]) if barcodes else 0.0
        
        return results
    
    def _extract_with_paddleocr(self, image: np.ndarray, results: dict) -> dict:
        """Extract using PaddleOCR"""
        # Convert BGR to RGB for PaddleOCR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run OCR
        ocr_result = self.engine.ocr(rgb_image, cls=True)
        
        barcodes = []
        text_blocks = []
        
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = float(line[1][1])
                    
                    if text.strip():
                        # Get bounding box
                        bbox = line[0]
                        flat_bbox = [coord for point in bbox for coord in point]
                        
                        text_blocks.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'bbox': tuple(flat_bbox)
                        })
                        
                        # Check for barcode patterns
                        barcode = self._extract_barcode_pattern(text)
                        if barcode:
                            barcodes.append({
                                'barcode': barcode,
                                'confidence': confidence,
                                'source_text': text,
                                'bbox': tuple(flat_bbox)
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

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR results"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply noise reduction
    denoised = cv2.medianBlur(gray, 3)
    
    # Apply thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up image
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return processed

def draw_ocr_results(image: np.ndarray, ocr_results: dict) -> np.ndarray:
    """Draw OCR results on image"""
    display_image = image.copy()
    
    # Draw bounding boxes for all text blocks
    for block in ocr_results['text_blocks']:
        bbox = block['bbox']
        if len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw barcode bounding boxes in red
    for barcode in ocr_results['barcodes']:
        bbox = barcode['bbox']
        if len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw barcode text
            label = f"Barcode: {barcode['barcode']}"
            cv2.putText(display_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return display_image

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
        ["pytesseract", "paddleocr"],
        help="Choose OCR engine for barcode extraction"
    )
    
    # Custom model upload
    st.sidebar.subheader("Upload Custom OCR Model")
    uploaded_model = st.sidebar.file_uploader(
        "Upload trained OCR model",
        type=['h5', 'pth', 'pt', 'onnx', 'pb', 'traineddata'],
        help="Upload custom trained OCR model files"
    )
    
    if uploaded_model is not None:
        if st.sidebar.button("Load Custom Model"):
            with st.spinner("Loading custom model..."):
                success = st.session_state.ocr_engine.load_custom_model(uploaded_model, engine_type)
                if success:
                    st.sidebar.success(f"Model loaded: {uploaded_model.name}")
    
    # Model info
    if st.session_state.ocr_engine.custom_model_loaded:
        st.sidebar.info(f"🔧 Using custom {engine_type} model")
    else:
        st.sidebar.info("🔧 Using default OCR engine")
    
    # Processing options
    st.sidebar.subheader("Processing Options")
    enable_preprocessing = st.sidebar.checkbox("Enable Image Preprocessing", value=True)
    show_text_blocks = st.sidebar.checkbox("Show All Text Blocks", value=False)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Images")
        
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
            if st.button("🚀 Extract Barcodes from All Images", type="primary"):
                process_multiple_images(uploaded_files, enable_preprocessing, show_text_blocks)
        
        # Single image processing
        if uploaded_files and len(uploaded_files) == 1:
            st.subheader("Single Image Preview")
            process_single_image(uploaded_files[0], enable_preprocessing, show_text_blocks)
    
    with col2:
        st.subheader("Results & Export")
        
        # Display results from session state
        if 'ocr_results' in st.session_state:
            display_results(st.session_state.ocr_results, show_text_blocks)
        else:
            st.info("Upload images and click 'Extract Barcodes' to see results")

def process_single_image(uploaded_file, enable_preprocessing: bool, show_text_blocks: bool):
    """Process a single uploaded image"""
    try:
        # Read image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert to BGR if needed
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Display original image
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Preprocess if enabled
        if enable_preprocessing:
            processed_image = preprocess_image(image_np)
            st.image(processed_image, caption="Preprocessed Image", use_column_width=True)
            image_to_process = processed_image
        else:
            image_to_process = image_np
        
        # Extract barcodes
        with st.spinner("Extracting barcodes..."):
            ocr_results = st.session_state.ocr_engine.extract_barcodes(image_to_process)
        
        # Store results
        st.session_state.ocr_results = {
            'filename': uploaded_file.name,
            'timestamp': datetime.now().isoformat(),
            'results': ocr_results
        }
        
        # Display results
        display_single_results(ocr_results, image_np, show_text_blocks)
        
    except Exception as e:
        st.error(f"Error processing image: {e}")

def process_multiple_images(uploaded_files, enable_preprocessing: bool, show_text_blocks: bool):
    """Process multiple uploaded images"""
    all_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        try:
            # Read and process image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Preprocess if enabled
            if enable_preprocessing:
                image_to_process = preprocess_image(image_np)
            else:
                image_to_process = image_np
            
            # Extract barcodes
            ocr_results = st.session_state.ocr_engine.extract_barcodes(image_to_process)
            
            # Store results
            all_results[uploaded_file.name] = {
                'timestamp': datetime.now().isoformat(),
                'results': ocr_results,
                'barcodes_found': len(ocr_results['barcodes'])
            }
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            all_results[uploaded_file.name] = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'barcodes_found': 0
            }
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Processing complete!")
    st.session_state.ocr_results = all_results
    
    # Show summary
    total_barcodes = sum(result.get('barcodes_found', 0) for result in all_results.values())
    st.success(f"✅ Processed {len(uploaded_files)} images, found {total_barcodes} barcodes")

def display_single_results(ocr_results: dict, original_image: np.ndarray, show_text_blocks: bool):
    """Display results for a single image"""
    st.subheader("Extraction Results")
    
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
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        st.image(result_image_rgb, caption="Barcode Detection Results", use_column_width=True)
    
    else:
        st.warning("❌ No barcodes found")
    
    # Show all text blocks if requested
    if show_text_blocks and ocr_results['text_blocks']:
        st.subheader("All Detected Text Blocks")
        text_data = []
        for block in ocr_results['text_blocks']:
            text_data.append({
                'Text': block['text'],
                'Confidence': f"{block['confidence']:.2%}",
                'Is Barcode': '✅' if any(b['source_text'] == block['text'] for b in ocr_results['barcodes']) else '❌'
            })
        
        st.table(text_data)

def display_results(all_results: dict, show_text_blocks: bool):
    """Display results for multiple images"""
    if isinstance(all_results, dict) and 'filename' in all_results:
        # Single image results
        display_single_results(all_results['results'], None, show_text_blocks)
    else:
        # Multiple image results
        st.subheader("Batch Processing Results")
        
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
        
        # Export results
        if st.button("📥 Export All Results as JSON"):
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'ocr_engine': st.session_state.ocr_engine.engine_type,
                'custom_model_used': st.session_state.ocr_engine.custom_model_loaded,
                'results': all_results
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"barcode_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Show detailed results for each file
        st.subheader("Detailed Results")
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
