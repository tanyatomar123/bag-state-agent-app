import streamlit as st
import cv2
import numpy as np
import time
import logging
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from PIL import Image
import pandas as pd
import plotly.express as px
from datetime import datetime
import random

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
    """Lightweight OCR simulator for demo purposes"""
    
    def __init__(self):
        self.simulated_barcodes = [
            "123456789012", "987654321098", "456123789045",
            "321654987012", "789012345678", "210987654321",
            "345678901234", "876543210987", "234567890123"
        ]
        st.success("✅ OCR Simulator loaded successfully")
    
    def process_image(self, image: np.ndarray, bbox: Optional[Tuple] = None) -> OCRResult:
        """Simulate OCR processing"""
        try:
            # Simulate OCR processing with random success
            if random.random() > 0.6:  # 40% chance to detect barcode
                text = random.choice(self.simulated_barcodes)
                confidence = random.uniform(0.7, 0.95)
            else:
                text = ""
                confidence = 0.0
            
            return OCRResult(
                text=text, 
                confidence=confidence, 
                bbox=bbox or (0, 0, image.shape[1], image.shape[0])
            )
            
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
        
        # Initialize OCR simulator
        self.ocr_model = OCRModel()
        
        logger.info("BagDetectionAgent initialized successfully")
    
    def simulate_bag_detection(self, image: np.ndarray) -> List[DetectionResult]:
        """Simulate bag detection for demo purposes"""
        height, width = image.shape[:2]
        detections = []
        
        # Randomly generate 0-2 bags per frame
        num_bags = random.randint(0, 2)
        
        for i in range(num_bags):
            # Generate random bounding box
            w, h = random.randint(100, 300), random.randint(150, 400)
            x1 = random.randint(50, width - w - 50)
            y1 = random.randint(50, height - h - 50)
            
            detections.append(DetectionResult(
                bbox=(x1, y1, x1 + w, y1 + h),
                confidence=random.uniform(0.7, 0.95),
                label="bag",
                class_id=0
            ))
        
        return detections
    
    def extract_barcode_from_bag(self, image: np.ndarray, bag_bbox: Tuple) -> Optional[str]:
        """Extract barcode from detected bag region"""
        try:
            x1, y1, x2, y2 = bag_bbox
            
            # Extract text from barcode region using OCR simulator
            ocr_result = self.ocr_model.process_image(image, bag_bbox)
            
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
            self.state_history.append({
                'old_state': old_state.name,
                'new_state': new_state.name,
                'timestamp': time.time()
            })
    
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
                self._handle_idle_state(has_bag)
                    
            elif self.state == AgentState.WAITING_FIRST_BAG:
                self._handle_waiting_first_bag_state(frame, has_bag, bag_detections)
                    
            elif self.state == AgentState.PROCESSING_BAG:
                self._handle_processing_bag_state(has_bag)
                    
            elif self.state == AgentState.WAITING_LAST_BAG:
                self._handle_waiting_last_bag_state(has_bag)
            
            # Check for timeout
            if (self.state in [AgentState.PROCESSING_BAG, AgentState.WAITING_LAST_BAG] and
                time.time() - self.processing_start_time > self.config['max_processing_time']):
                self.change_state(AgentState.ERROR)
                
            return self._get_current_status(bag_detections)
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.change_state(AgentState.ERROR)
            return self._get_current_status([])
    
    def _handle_idle_state(self, has_bag: bool):
        """Handle IDLE state transitions"""
        if has_bag:
            self.bag_present_count += 1
            if self.bag_present_count >= self.config['bag_presence_frames']:
                self.change_state(AgentState.WAITING_FIRST_BAG)
                self.bag_present_count = 0
        else:
            self.bag_present_count = 0
    
    def _handle_waiting_first_bag_state(self, frame: np.ndarray, has_bag: bool, detections: List[DetectionResult]):
        """Handle WAITING_FIRST_BAG state transitions"""
        if has_bag and not self.first_bag_detected:
            if detections:
                primary_bag = detections[0]
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
    
    def _handle_processing_bag_state(self, has_bag: bool):
        """Handle PROCESSING_BAG state transitions"""
        if not has_bag:
            self.bag_absent_count += 1
            if self.bag_absent_count >= self.config['bag_absence_frames']:
                self.change_state(AgentState.WAITING_LAST_BAG)
                self.bag_absent_count = 0
        else:
            self.bag_absent_count = 0
    
    def _handle_waiting_last_bag_state(self, has_bag: bool):
        """Handle WAITING_LAST_BAG state transitions"""
        if has_bag:
            self.bag_present_count += 1
            if self.bag_present_count >= self.config['bag_presence_frames']:
                self.last_bag_detected = True
                self.change_state(AgentState.COMPLETED)
        else:
            self.bag_present_count = 0
    
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
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = BagDetectionAgent()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    # Sidebar Configuration
    st.sidebar.header("Configuration")
    
    # Detection Parameters
    st.sidebar.subheader("Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    ocr_confidence = st.sidebar.slider("OCR Confidence", 0.1, 1.0, 0.6)
    presence_frames = st.sidebar.slider("Bag Presence Frames", 1, 10, 3)
    absence_frames = st.sidebar.slider("Bag Absence Frames", 1, 15, 5)
    max_processing_time = st.sidebar.slider("Max Processing Time (s)", 10, 60, 30)
    
    # Update agent config
    st.session_state.agent.config.update({
        'confidence_threshold': confidence_threshold,
        'ocr_confidence_threshold': ocr_confidence,
        'bag_presence_frames': presence_frames,
        'bag_absence_frames': absence_frames,
        'max_processing_time': max_processing_time,
    })
    
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
                st.session_state.frame_count = 0
                
        with control_col2:
            if st.button("⏹️ Stop Processing", use_container_width=True):
                st.session_state.processing = False
                
        with control_col3:
            if st.button("🔄 Reset Agent", use_container_width=True):
                st.session_state.agent.reset()
                st.session_state.processing = False
                st.session_state.frame_count = 0
        
        # Video feed placeholder
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Process frames if active
        if st.session_state.processing:
            # Create sample frame with some visual variation
            st.session_state.frame_count += 1
            frame = create_sample_frame(st.session_state.frame_count)
            
            # Process frame
            status = st.session_state.agent.process_frame(frame)
            
            # Draw detections on frame
            display_frame = draw_detections(frame, status['detections'])
            
            # Convert to PIL for display
            display_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            video_placeholder.image(display_image, caption="Live Feed with Detections", use_column_width=True)
            
            # Display status
            display_status(status, status_placeholder)
            
            # Auto-reset if completed
            if status['state'] == 'COMPLETED':
                time.sleep(2)  # Show completed state for 2 seconds
                st.session_state.agent.reset()
                st.rerun()
                
        else:
            # Show static image when not processing
            sample_image = create_sample_frame(0)
            sample_image = Image.fromarray(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
            video_placeholder.image(sample_image, caption="Ready to Start Processing", use_column_width=True)
            status_placeholder.info("Click 'Start Processing' to begin bag detection")
    
    with col2:
        st.subheader("Agent Statistics")
        display_statistics()

    # Footer
    st.markdown("---")
    st.markdown("### 📊 System Information")
    display_system_info()

def create_sample_frame(frame_count: int) -> np.ndarray:
    """Create a sample frame with some visual variation"""
    # Create a base frame
    frame = np.random.randint(50, 100, (480, 640, 3), dtype=np.uint8)
    
    # Add some visual elements that change over time
    cv2.rectangle(frame, (100, 100), (540, 380), (200, 200, 200), 2)
    cv2.putText(frame, "Conveyor Belt View", (150, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add moving elements for visual interest
    x_pos = (frame_count * 5) % 600
    cv2.circle(frame, (x_pos + 20, 240), 15, (0, 255, 255), -1)
    
    return frame

def draw_detections(frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    """Draw detection bounding boxes on frame"""
    display_frame = frame.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = (0, 255, 0)  # Green for bags
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"Bag: {detection.confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(display_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return display_frame

def display_status(status: Dict[str, Any], placeholder):
    """Display current processing status"""
    with placeholder.container():
        st.subheader("Current Status")
        
        # State indicator with colors
        state_colors = {
            "IDLE": "blue",
            "WAITING_FIRST_BAG": "yellow",
            "PROCESSING_BAG": "orange", 
            "WAITING_LAST_BAG": "purple",
            "COMPLETED": "green",
            "ERROR": "red"
        }
        
        state_color = state_colors.get(status['state'], "gray")
        st.markdown(f"**State:** <span style='color: {state_color}; font-weight: bold; font-size: 1.2em'>{status['state']}</span>", 
                   unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            barcode_text = status['barcode_data'] or "None"
            st.metric("Barcode Found", barcode_text)
        with col2:
            st.metric("Processing Time", f"{status['processing_time']:.1f}s")
        with col3:
            st.metric("State Duration", f"{status['state_duration']:.1f}s")
        
        # Progress bar
        if status['state'] in ['PROCESSING_BAG', 'WAITING_LAST_BAG']:
            progress = min(status['processing_time'] / 30.0, 1.0)
            st.progress(progress, text=f"Processing: {progress*100:.1f}%")
        
        # Additional info
        if status['state'] == 'COMPLETED':
            st.success("🎉 Processing completed successfully!")
        elif status['state'] == 'ERROR':
            st.error("❌ Processing error occurred")

def display_statistics():
    """Display agent statistics and charts"""
    agent = st.session_state.agent
    
    if st.session_state.processing:
        current_status = agent._get_current_status([])
        
        # State history chart
        if agent.state_history:
            df_states = pd.DataFrame(agent.state_history)
            df_states['time'] = pd.to_datetime(df_states['timestamp'], unit='s')
            
            fig = px.timeline(df_states, x_start="time", y="new_state", color="new_state",
                             title="State Transition History", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detection history
        if agent.detection_history:
            df_detections = pd.DataFrame(agent.detection_history[-30:])  # Last 30 frames
            df_detections['time'] = pd.to_datetime(df_detections['timestamp'], unit='s')
            
            fig = px.line(df_detections, x='time', y='detections', 
                         title="Bag Detections Over Time", height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Results summary
        st.subheader("Processing Results")
        if current_status['barcode_data']:
            st.success(f"✅ Barcode: **{current_status['barcode_data']}**")
        else:
            st.warning("⏳ No barcode detected yet")
            
        st.metric("Total State Changes", len(agent.state_history))
        st.metric("Frames Processed", len(agent.detection_history))
        
        # State distribution
        if agent.detection_history:
            state_counts = pd.DataFrame(agent.detection_history)['state'].value_counts()
            fig = px.pie(values=state_counts.values, names=state_counts.index,
                        title="State Distribution", height=250)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Start processing to see statistics")

def display_system_info():
    """Display system information footer"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("OCR Backend", "Simulator")
    
    with col2:
        status = "Active" if st.session_state.processing else "Idle"
        st.metric("State Machine", status)
    
    with col3:
        frames = len(st.session_state.agent.detection_history) if st.session_state.processing else 0
        st.metric("Frames Processed", frames)
    
    with col4:
        states = len(st.session_state.agent.state_history)
        st.metric("State Transitions", states)

if __name__ == "__main__":
    main()
