import streamlit as st
import cv2
import numpy as np
import time
import logging
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import tempfile
import os
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Bag Detection AI Agent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AgentState(Enum):
    IDLE = auto()
    WAITING_FIRST_BAG = auto()
    PROCESSING_BAG = auto()
    WAITING_LAST_BAG = auto()
    COMPLETED = auto()
    ERROR = auto()

@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str
    class_id: int

@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]

class OCRModel:
    """OCR model wrapper with multiple backend options"""
    
    def __init__(self, model_type: str = 'paddle', model_path: Optional[str] = None):
        self.model_type = model_type
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path: Optional[str] = None):
        """Load OCR model based on type"""
        try:
            if self.model_type == 'paddle':
                try:
                    import paddleocr
                    self.model = paddleocr.PaddleOCR(
                        use_angle_cls=True,
                        lang='en',
                        show_log=False
                    )
                    st.success("✅ PaddleOCR model loaded successfully")
                except ImportError:
                    st.error("❌ PaddleOCR not installed. Using fallback OCR.")
                    self.model_type = 'fallback'
                    
            elif self.model_type == 'easyocr':
                try:
                    import easyocr
                    self.model = easyocr.Reader(['en'])
                    st.success("✅ EasyOCR model loaded successfully")
                except ImportError:
                    st.error("❌ EasyOCR not installed. Using fallback OCR.")
                    self.model_type = 'fallback'
                    
            elif self.model_type == 'tesseract':
                try:
                    import pytesseract
                    self.model = pytesseract
                    st.success("✅ Tesseract OCR ready")
                except ImportError:
                    st.error("❌ Tesseract not installed. Using fallback OCR.")
                    self.model_type = 'fallback'
            
            # Fallback OCR using basic text detection
            if self.model_type == 'fallback':
                st.info("Using fallback OCR (simulated)")
                self.model = None
                
        except Exception as e:
            st.error(f"❌ Failed to load OCR model: {e}")
            self.model_type = 'fallback'
            self.model = None
    
    def process_image(self, image: np.ndarray, bbox: Optional[Tuple] = None) -> OCRResult:
        """Extract text from image or ROI"""
        try:
            if bbox:
                x1, y1, x2, y2 = bbox
                roi = image[y1:y2, x1:x2]
            else:
                roi = image
            
            # For demo purposes, simulate OCR with random barcodes
            if self.model_type == 'fallback' or self.model is None:
                # Simulate OCR processing for demo
                simulated_barcodes = [
                    "123456789012", "987654321098", "456123789045",
                    "321654987012", "789012345678", "210987654321"
                ]
                import random
                if random.random() > 0.7:  # 30% chance to detect barcode
                    text = random.choice(simulated_barcodes)
                    confidence = random.uniform(0.7, 0.95)
                else:
                    text = ""
                    confidence = 0.0
                return OCRResult(text=text, confidence=confidence, bbox=bbox or (0, 0, image.shape[1], image.shape[0]))
            
            if self.model_type == 'paddle':
                result = self.model.ocr(roi, cls=True)
                if result and result[0]:
                    text = result[0][0][1][0]
                    confidence = result[0][0][1][1]
                    return OCRResult(text=text, confidence=confidence, bbox=bbox or (0, 0, image.shape[1], image.shape[0]))
            
            elif self.model_type == 'easyocr':
                results = self.model.readtext(roi)
                if results:
                    text = results[0][1]
                    confidence = results[0][2]
                    return OCRResult(text=text, confidence=confidence, bbox=bbox or (0, 0, image.shape[1], image.shape[0]))
            
            elif self.model_type == 'tesseract':
                text = self.model.image_to_string(roi)
                return OCRResult(text=text.strip(), confidence=0.8, bbox=bbox or (0, 0, image.shape[1], image.shape[0]))
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
        
        return OCRResult(text="", confidence=0.0, bbox=bbox or (0, 0, image.shape[1], image.shape[0]))

class BagDetectionAgent:
    """Lightweight Bag Detection Agent for Streamlit Demo"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'confidence_threshold': 0.5,
            'ocr_confidence_threshold': 0.6,
            'max_processing_time': 30.0,
            'bag_presence_frames': 3,
            'bag_absence_frames': 5,
            'barcode_region_ratio': 0.3,
        }
        if config:
            self.config.update(config)
        
        # State management
        self.state = AgentState.IDLE
        self.state_history = []
        self.last_state_change = time.time()
        
        # Detection counters
        self.bag_present_count = 0
        self.bag_absent_count = 0
        self.processing_start_time = 0
        
        # Results storage
        self.first_bag_detected = False
        self.barcode_data = None
        self.last_bag_detected = False
        self.detection_history = []
        
        # Initialize OCR (using fallback for demo)
        self.ocr_model = OCRModel('fallback')
        
        logger.info("BagDetectionAgent initialized successfully")
    
    def simulate_bag_detection(self, image: np.ndarray) -> List[DetectionResult]:
        """Simulate bag detection for demo purposes"""
        # For demo, randomly detect bags with some probability
        import random
        
        if random.random() > 0.3:  # 70% chance to detect a bag
            height, width = image.shape[:2]
            # Generate random bounding box
            w, h = random.randint(100, 300), random.randint(150, 400)
            x1 = random.randint(50, width - w - 50)
            y1 = random.randint(50, height - h - 50)
            
            return [DetectionResult(
                bbox=(x1, y1, x1 + w, y1 + h),
                confidence=random.uniform(0.7, 0.95),
                label="bag",
                class_id=0
            )]
        return []
    
    def extract_barcode_from_bag(self, image: np.ndarray, bag_bbox: Tuple) -> Optional[str]:
        """Extract barcode from detected bag region"""
        try:
            x1, y1, x2, y2 = bag_bbox
            bag_height = y2 - y1
            
            # Assume barcode is in bottom portion of bag
            barcode_region_height = int(bag_height * self.config['barcode_region_ratio'])
            barcode_y1 = y2 - barcode_region_height
            barcode_bbox = (x1, barcode_y1, x2, y2)
            
            # Extract text from barcode region
            ocr_result = self.ocr_model.process_image(image, barcode_bbox)
            
            if (ocr_result.confidence >= self.config['ocr_confidence_threshold'] and 
                ocr_result.text.strip()):
                
                barcode_text = ocr_result.text.strip()
                if self._validate_barcode(barcode_text):
                    logger.info(f"Barcode extracted: {barcode_text}")
                    return barcode_text
            
            return None
            
        except Exception as e:
            logger.error(f"Barcode extraction failed: {e}")
            return None
    
    def _validate_barcode(self, barcode: str) -> bool:
        """Basic barcode validation"""
        cleaned_barcode = ''.join(c for c in barcode if c.isalnum())
        return len(cleaned_barcode) >= 6
    
    def change_state(self, new_state: AgentState):
        """Safely change agent state"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self.last_state_change = time.time()
            self.state_history.append((old_state, new_state, time.time()))
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Main processing method for each frame"""
        try:
            # Simulate bag detection
            bag_detections = self.simulate_bag_detection(frame)
            has_bag = len(bag_detections) > 0
            
            # Record detection for visualization
            self.detection_history.append({
                'timestamp': time.time(),
                'has_bag': has_bag,
                'detections': len(bag_detections),
                'state': self.state.name
            })
            
            # State machine logic
            if self.state == AgentState.IDLE:
                if has_bag:
                    self.bag_present_count += 1
                    if self.bag_present_count >= self.config['bag_presence_frames']:
                        self.change_state(AgentState.WAITING_FIRST_BAG)
                        self.bag_present_count = 0
                else:
                    self.bag_present_count = 0
                    
            elif self.state == AgentState.WAITING_FIRST_BAG:
                if has_bag and not self.first_bag_detected:
                    primary_bag = bag_detections[0] if bag_detections else None
                    if primary_bag:
                        barcode = self.extract_barcode_from_bag(frame, primary_bag.bbox)
                        if barcode:
                            self.barcode_data = barcode
                            self.first_bag_detected = True
                            self.processing_start_time = time.time()
                            self.change_state(AgentState.PROCESSING_BAG)
                else:
                    self.bag_absent_count += 1
                    if self.bag_absent_count >= self.config['bag_absence_frames']:
                        self.change_state(AgentState.IDLE)
                        self.bag_absent_count = 0
                        
            elif self.state == AgentState.PROCESSING_BAG:
                if not has_bag:
                    self.bag_absent_count += 1
                    if self.bag_absent_count >= self.config['bag_absence_frames']:
                        self.change_state(AgentState.WAITING_LAST_BAG)
                        self.bag_absent_count = 0
                else:
                    self.bag_absent_count = 0
                    
            elif self.state == AgentState.WAITING_LAST_BAG:
                if has_bag:
                    self.bag_present_count += 1
                    if self.bag_present_count >= self.config['bag_presence_frames']:
                        self.last_bag_detected = True
                        self.change_state(AgentState.COMPLETED)
                else:
                    self.bag_present_count = 0
            
            # Check for timeout
            if (self.state in [AgentState.PROCESSING_BAG, AgentState.WAITING_LAST_BAG] and
                time.time() - self.processing_start_time > self.config['max_processing_time']):
                self.change_state(AgentState.ERROR)
                
            return self._get_current_status(bag_detections)
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.change_state(AgentState.ERROR)
            return self._get_current_status([])
    
    def _get_current_status(self, detections: List[DetectionResult]) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'state': self.state.name,
            'first_bag_detected': self.first_bag_detected,
            'last_bag_detected': self.last_bag_detected,
            'barcode_data': self.barcode_data,
            'detections': detections,
            'processing_time': time.time() - self.processing_start_time if self.processing_start_time else 0,
            'state_duration': time.time() - self.last_state_change
        }
    
    def reset(self):
        """Reset agent to initial state"""
        self.state = AgentState.IDLE
        self.first_bag_detected = False
        self.last_bag_detected = False
        self.barcode_data = None
        self.bag_present_count = 0
        self.bag_absent_count = 0
        self.processing_start_time = 0
        self.detection_history.clear()

# Streamlit App
def main():
    st.title("🛍️ Bag Detection AI Agent")
    st.markdown("Lightweight production-ready state machine for bag detection and barcode extraction")
    
    # Sidebar Configuration
    st.sidebar.header("Configuration")
    
    # OCR Model Selection
    ocr_model = st.sidebar.selectbox(
        "OCR Model",
        ["PaddleOCR", "EasyOCR", "Tesseract", "Fallback (Demo)"],
        index=3
    )
    
    # Detection Parameters
    st.sidebar.subheader("Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    ocr_confidence = st.sidebar.slider("OCR Confidence", 0.1, 1.0, 0.6)
    presence_frames = st.sidebar.slider("Bag Presence Frames", 1, 10, 3)
    absence_frames = st.sidebar.slider("Bag Absence Frames", 1, 15, 5)
    
    # Initialize agent with configuration
    config = {
        'confidence_threshold': confidence_threshold,
        'ocr_confidence_threshold': ocr_confidence,
        'bag_presence_frames': presence_frames,
        'bag_absence_frames': absence_frames,
    }
    
    if 'agent' not in st.session_state:
        st.session_state.agent = BagDetectionAgent(config)
    
    # Update agent config
    st.session_state.agent.config.update(config)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Processing")
        
        # Processing controls
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            if st.button("🎬 Start Processing", use_container_width=True):
                st.session_state.processing = True
                st.session_state.agent.reset()
                
        with control_col2:
            if st.button("⏹️ Stop Processing", use_container_width=True):
                st.session_state.processing = False
                
        with control_col3:
            if st.button("🔄 Reset Agent", use_container_width=True):
                st.session_state.agent.reset()
                st.session_state.processing = False
        
        # Video feed placeholder
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Process frames if active
        if st.session_state.get('processing', False):
            # Create sample frame (in real app, this would come from Stapipy)
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Process frame
            status = st.session_state.agent.process_frame(frame)
            
            # Draw detections on frame
            display_frame = frame.copy()
            for detection in status['detections']:
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Bag: {detection.confidence:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert to PIL for display
            display_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            video_placeholder.image(display_image, caption="Live Feed with Detections", use_column_width=True)
            
            # Display status
            with status_placeholder.container():
                st.subheader("Current Status")
                
                # State indicator
                state_colors = {
                    "IDLE": "blue",
                    "WAITING_FIRST_BAG": "yellow",
                    "PROCESSING_BAG": "orange", 
                    "WAITING_LAST_BAG": "purple",
                    "COMPLETED": "green",
                    "ERROR": "red"
                }
                
                state_color = state_colors.get(status['state'], "gray")
                st.markdown(f"**State:** <span style='color: {state_color}; font-weight: bold'>{status['state']}</span>", 
                           unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Barcode Found", status['barcode_data'] or "None")
                with col2:
                    st.metric("Processing Time", f"{status['processing_time']:.1f}s")
                with col3:
                    st.metric("State Duration", f"{status['state_duration']:.1f}s")
                
                # Progress
                if status['state'] in ['PROCESSING_BAG', 'WAITING_LAST_BAG']:
                    progress = min(status['processing_time'] / 30.0, 1.0)
                    st.progress(progress, text=f"Processing: {progress*100:.1f}%")
        
        else:
            # Show static image when not processing
            sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            sample_image = Image.fromarray(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
            video_placeholder.image(sample_image, caption="Ready to Start Processing", use_column_width=True)
            status_placeholder.info("Click 'Start Processing' to begin bag detection")
    
    with col2:
        st.subheader("Agent Statistics")
        
        # Current status
        if st.session_state.get('processing', False):
            current_status = st.session_state.agent._get_current_status([])
            
            # State history chart
            if st.session_state.agent.state_history:
                states_data = []
                for old_state, new_state, timestamp in st.session_state.agent.state_history:
                    states_data.append({
                        'timestamp': datetime.fromtimestamp(timestamp),
                        'state': new_state.name,
                        'transition': f"{old_state.name} → {new_state.name}"
                    })
                
                if states_data:
                    df_states = pd.DataFrame(states_data)
                    fig = px.timeline(df_states, x_start="timestamp", y="state", color="state",
                                     title="State Transition History")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detection history
            if st.session_state.agent.detection_history:
                df_detections = pd.DataFrame(st.session_state.agent.detection_history[-50:])  # Last 50 frames
                fig = px.line(df_detections, x='timestamp', y='detections', 
                             title="Bag Detections Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            # Results summary
            st.subheader("Processing Results")
            if current_status['barcode_data']:
                st.success(f"✅ Barcode: {current_status['barcode_data']}")
            else:
                st.warning("⏳ No barcode detected yet")
                
            st.metric("Total State Changes", len(st.session_state.agent.state_history))
            st.metric("Detection History Frames", len(st.session_state.agent.detection_history))
        
        else:
            st.info("Start processing to see statistics")
        
        # Configuration summary
        st.subheader("Current Configuration")
        config_df = pd.DataFrame([
            {"Parameter": "OCR Model", "Value": ocr_model},
            {"Parameter": "Confidence Threshold", "Value": confidence_threshold},
            {"Parameter": "OCR Confidence", "Value": ocr_confidence},
            {"Parameter": "Presence Frames", "Value": presence_frames},
            {"Parameter": "Absence Frames", "Value": absence_frames},
        ])
        st.dataframe(config_df, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    st.markdown("### 📊 System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("OCR Backend", ocr_model)
    
    with col2:
        st.metric("State Machine", "Active" if st.session_state.get('processing', False) else "Idle")
    
    with col3:
        if st.session_state.get('processing', False):
            st.metric("Frames Processed", len(st.session_state.agent.detection_history))
        else:
            st.metric("Frames Processed", "0")

if __name__ == "__main__":
    main()
