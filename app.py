import streamlit as st
import cv2
import numpy as np
import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd
import plotly.express as px
from PIL import Image
import json
import tempfile
import os
from pathlib import Path
import asyncio
import queue
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Bag Detection Pipeline Agent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AgentState(Enum):
    """State machine states for the bag detection pipeline"""
    IDLE = auto()
    WAITING_FIRST_BAG = auto()
    FIRST_BAG_DETECTED = auto()
    PROCESSING_FIRST_OCR = auto()
    TRACKING_BAGS = auto()
    WAITING_LAST_BAG = auto()
    LAST_BAG_DETECTED = auto()
    PROCESSING_LAST_OCR = auto()
    COMPLETED = auto()
    ERROR = auto()

@dataclass
class BagDetection:
    """Bag detection result from YOLO"""
    bag_id: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    timestamp: datetime
    frame_id: int
    class_id: int = 0
    class_name: str = "bag"

@dataclass
class OCRResult:
    """OCR extraction result"""
    barcode: str
    confidence: float
    timestamp: datetime
    bag_id: str
    bbox: Tuple[int, int, int, int]
    is_first_bag: bool = False
    is_last_bag: bool = False

@dataclass
class PipelineContext:
    """Context for the entire pipeline"""
    first_bag: Optional[BagDetection] = None
    last_bag: Optional[BagDetection] = None
    first_barcode: Optional[str] = None
    last_barcode: Optional[str] = None
    all_detections: List[BagDetection] = field(default_factory=list)
    ocr_results: List[OCRResult] = field(default_factory=list)
    total_bags: int = 0
    frames_without_bag: int = 0
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    error_message: Optional[str] = None
    pipeline_start_time: Optional[datetime] = None

class StapipySimulator:
    """Simulates Stapipy frame grabbing"""
    
    def __init__(self, source_type: str = "webcam", video_path: Optional[str] = None):
        self.source_type = source_type
        self.video_path = video_path
        self.frame_count = 0
        self.cap = None
        self._initialize_source()
    
    def _initialize_source(self):
        """Initialize video source"""
        try:
            if self.source_type == "webcam":
                self.cap = cv2.VideoCapture(0)
            elif self.source_type == "video_file" and self.video_path:
                self.cap = cv2.VideoCapture(self.video_path)
            else:
                # Create synthetic frames
                self.cap = None
        except Exception as e:
            logger.warning(f"Could not initialize video source: {e}")
            self.cap = None
    
    def get_frame(self) -> Optional[Dict]:
        """Get next frame from source"""
        try:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.frame_count += 1
                    return {
                        'frame': frame,
                        'frame_id': self.frame_count,
                        'timestamp': datetime.now(),
                        'source': 'stapipy'
                    }
                else:
                    # Loop video or restart webcam
                    if self.source_type == "video_file":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        return self.get_frame()
                    return None
            else:
                # Generate synthetic frame
                return self._generate_synthetic_frame()
                
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None
    
    def _generate_synthetic_frame(self) -> Dict:
        """Generate synthetic conveyor belt frame"""
        self.frame_count += 1
        
        # Create a realistic conveyor belt frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:,:] = (50, 50, 50)  # Dark gray background
        
        # Draw conveyor belt
        belt_y1, belt_y2 = 150, 330
        cv2.rectangle(frame, (50, belt_y1), (590, belt_y2), (100, 100, 100), -1)
        cv2.rectangle(frame, (50, belt_y1), (590, belt_y2), (150, 150, 150), 2)
        
        # Draw moving elements based on frame count
        x_pos = (self.frame_count * 3) % 600
        
        # Occasionally add bags
        if random.random() > 0.4:  # 60% chance of bag
            bag_width, bag_height = 80, 120
            bag_x = 50 + x_pos
            bag_y = belt_y1 + 10
            
            # Draw bag
            cv2.rectangle(frame, (bag_x, bag_y), (bag_x + bag_width, bag_y + bag_height), 
                         (0, 100, 200), -1)
            cv2.rectangle(frame, (bag_x, bag_y), (bag_x + bag_width, bag_y + bag_height), 
                         (255, 255, 255), 2)
            
            # Draw barcode area
            barcode_y = bag_y + bag_height - 25
            cv2.rectangle(frame, (bag_x + 10, barcode_y), (bag_x + bag_width - 10, bag_y + bag_height - 5), 
                         (255, 255, 255), -1)
            # Simulate barcode lines
            for i in range(bag_x + 15, bag_x + bag_width - 15, 5):
                if random.random() > 0.3:
                    cv2.line(frame, (i, barcode_y), (i, bag_y + bag_height - 10), (0, 0, 0), 1)
        
        # Add text
        cv2.putText(frame, "Conveyor Belt - Stapipy Stream", (150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 430), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return {
            'frame': frame,
            'frame_id': self.frame_count,
            'timestamp': datetime.now(),
            'source': 'synthetic'
        }
    
    def release(self):
        """Release resources"""
        if self.cap:
            self.cap.release()

class YOLODetector:
    """YOLO detection wrapper"""
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            st.success(f"✅ YOLO model loaded: {self.model_path}")
        except Exception as e:
            st.warning(f"⚠️ YOLO not available, using simulation: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in frame"""
        if self.model is None:
            return self._simulate_detection(frame)
        
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        class_name = result.names[cls] if hasattr(result, 'names') else "object"
                        
                        # Filter for bags/objects of interest
                        if class_name in ["bag", "suitcase", "backpack", "handbag"] or conf > 0.5:
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': conf,
                                'class_id': cls,
                                'class_name': class_name
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return self._simulate_detection(frame)
    
    def _simulate_detection(self, frame: np.ndarray) -> List[Dict]:
        """Simulate bag detection"""
        height, width = frame.shape[:2]
        detections = []
        
        # Look for bag-like rectangles in the conveyor area
        conveyor_region = frame[150:330, 50:590]  # Conveyor belt area
        
        # Simple color-based detection for blue bags
        blue_lower = np.array([100, 0, 0])
        blue_upper = np.array([255, 100, 100])
        blue_mask = cv2.inRange(conveyor_region, blue_lower, blue_upper)
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum bag size
                x, y, w, h = cv2.boundingRect(contour)
                # Convert to full frame coordinates
                x_full = x + 50
                y_full = y + 150
                
                detections.append({
                    'bbox': (x_full, y_full, x_full + w, y_full + h),
                    'confidence': min(0.9, area / 5000),  # Scale confidence by size
                    'class_id': 0,
                    'class_name': 'bag'
                })
        
        # If no contours found, use random detection for demo
        if not detections and random.random() > 0.6:
            w, h = 80, 120
            x = random.randint(50, width - w - 50)
            y = random.randint(150, 330 - h)
            
            detections.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': random.uniform(0.7, 0.95),
                'class_id': 0,
                'class_name': 'bag'
            })
        
        return detections

class OCREngine:
    """OCR engine for barcode extraction"""
    
    def __init__(self, engine_type: str = "paddle", model_path: Optional[str] = None):
        self.engine_type = engine_type
        self.model_path = model_path
        self.engine = None
        self._load_engine()
    
    def _load_engine(self):
        """Load OCR engine"""
        try:
            if self.engine_type == "paddle":
                from paddleocr import PaddleOCR
                self.engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                st.success("✅ PaddleOCR engine loaded")
            else:
                st.warning("Using OCR simulation")
                self.engine = None
        except Exception as e:
            st.warning(f"OCR engine not available: {e}")
            self.engine = None
    
    def extract_barcode(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Extract barcode from bag region"""
        if self.engine is None:
            return self._simulate_barcode_extraction()
        
        try:
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # Convert to RGB for OCR
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            if self.engine_type == "paddle":
                result = self.engine.ocr(roi_rgb, cls=True)
                
                if result and result[0]:
                    texts = [line[1][0] for line in result[0]]
                    barcode = self._find_barcode_pattern(texts)
                    if barcode:
                        return barcode
            
            return None
            
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return self._simulate_barcode_extraction()
    
    def _find_barcode_pattern(self, texts: List[str]) -> Optional[str]:
        """Find barcode patterns in OCR results"""
        import re
        
        barcode_patterns = [
            r'\b\d{12,13}\b',      # EAN-13, UPC
            r'\b\d{8}\b',          # EAN-8
            r'\b[A-Z0-9]{8,15}\b', # Alphanumeric codes
        ]
        
        for text in texts:
            for pattern in barcode_patterns:
                match = re.search(pattern, text.replace(' ', ''))
                if match:
                    return match.group(0)
        
        return None
    
    def _simulate_barcode_extraction(self) -> Optional[str]:
        """Simulate barcode extraction"""
        barcodes = [
            "5901234123457", "9780201379624", "1234567890128",
            "4006381333931", "3661112507010", "5449000000996",
            "3017620422003", "7613032620033", "8000500310427"
        ]
        
        # 70% success rate for simulation
        if random.random() > 0.3:
            return random.choice(barcodes)
        return None

class BagDetectionAgent:
    """
    Main agent that manages the complete pipeline:
    1. Stapipy frame grabbing
    2. YOLO bag detection  
    3. OCR barcode extraction
    4. State management for 1st/last bag detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = AgentState.IDLE
        self.context = PipelineContext()
        
        # Pipeline components
        self.stapipy = StapipySimulator(
            source_type=config.get('source_type', 'synthetic'),
            video_path=config.get('video_path')
        )
        self.yolo = YOLODetector(
            model_path=config.get('yolo_model', 'yolov8n.pt'),
            conf_threshold=config.get('confidence_threshold', 0.5)
        )
        self.ocr = OCREngine(
            engine_type=config.get('ocr_engine', 'paddle')
        )
        
        # State tracking
        self.current_frame = None
        self.processing_active = False
        self.state_history = []
        self.last_detection_time = None
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        
        logger.info(f"BagDetectionAgent initialized with session: {self.context.session_id}")
    
    def start_pipeline(self):
        """Start the processing pipeline"""
        self.state = AgentState.WAITING_FIRST_BAG
        self.context.pipeline_start_time = datetime.now()
        self.processing_active = True
        self.state_history.append(('START', self.state.name, datetime.now()))
        
        logger.info("Pipeline started - waiting for first bag")
    
    def stop_pipeline(self):
        """Stop the processing pipeline"""
        self.processing_active = False
        self.stapipy.release()
        logger.info("Pipeline stopped")
    
    def process_single_frame(self) -> bool:
        """
        Process a single frame through the complete pipeline
        Returns True if processing should continue
        """
        if not self.processing_active:
            return False
        
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # 1. Stapipy Frame Grabbing
            frame_data = self.stapipy.get_frame()
            if frame_data is None:
                logger.warning("No frame received from Stapipy")
                return True
            
            self.current_frame = frame_data['frame']
            
            # 2. YOLO Detection
            detections = self.yolo.detect(self.current_frame)
            
            # 3. State Management & OCR Processing
            has_bag = len(detections) > 0
            
            if self.state == AgentState.WAITING_FIRST_BAG:
                self._handle_waiting_first_bag(detections, frame_data)
                
            elif self.state == AgentState.FIRST_BAG_DETECTED:
                self._handle_first_bag_detected()
                
            elif self.state == AgentState.TRACKING_BAGS:
                self._handle_tracking_bags(detections, frame_data)
                
            elif self.state == AgentState.WAITING_LAST_BAG:
                self._handle_waiting_last_bag(has_bag)
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return self.state != AgentState.COMPLETED and self.state != AgentState.ERROR
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            self.context.error_message = str(e)
            self._change_state(AgentState.ERROR)
            return False
    
    def _handle_waiting_first_bag(self, detections: List[Dict], frame_data: Dict):
        """Handle state: Waiting for first bag"""
        if detections:
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            bag_detection = BagDetection(
                bag_id="BAG_001",
                confidence=best_detection['confidence'],
                bbox=best_detection['bbox'],
                timestamp=frame_data['timestamp'],
                frame_id=frame_data['frame_id']
            )
            
            self.context.first_bag = bag_detection
            self.context.all_detections.append(bag_detection)
            self.context.total_bags = 1
            self.last_detection_time = datetime.now()
            
            self._change_state(AgentState.FIRST_BAG_DETECTED)
            logger.info(f"First bag detected: {bag_detection.bag_id}")
            
            # Start OCR processing for first bag
            self._process_bag_ocr(bag_detection, is_first=True)
        
        else:
            self.context.frames_without_bag += 1
    
    def _handle_first_bag_detected(self):
        """Handle state: First bag detected - wait for OCR completion"""
        # Check if we have OCR result for first bag
        first_bag_ocr = next((ocr for ocr in self.context.ocr_results 
                            if ocr.is_first_bag), None)
        
        if first_bag_ocr:
            self.context.first_barcode = first_bag_ocr.barcode
            self._change_state(AgentState.TRACKING_BAGS)
            logger.info(f"First barcode extracted: {first_bag_ocr.barcode}")
    
    def _handle_tracking_bags(self, detections: List[Dict], frame_data: Dict):
        """Handle state: Tracking subsequent bags"""
        if detections:
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            bag_detection = BagDetection(
                bag_id=f"BAG_{self.context.total_bags + 1:03d}",
                confidence=best_detection['confidence'],
                bbox=best_detection['bbox'],
                timestamp=frame_data['timestamp'],
                frame_id=frame_data['frame_id']
            )
            
            self.context.all_detections.append(bag_detection)
            self.context.total_bags += 1
            self.last_detection_time = datetime.now()
            self.context.frames_without_bag = 0
            
            logger.info(f"Bag detected: {bag_detection.bag_id}")
            
            # Process OCR for this bag
            self._process_bag_ocr(bag_detection, is_first=False)
            
        else:
            self.context.frames_without_bag += 1
            
            # Check for last bag timeout
            if self.last_detection_time:
                timeout = self.config.get('last_bag_timeout', 5.0)
                elapsed = (datetime.now() - self.last_detection_time).total_seconds()
                
                if elapsed >= timeout:
                    self._change_state(AgentState.WAITING_LAST_BAG)
                    logger.info(f"Last bag timeout reached: {elapsed:.1f}s")
    
    def _handle_waiting_last_bag(self, has_bag: bool):
        """Handle state: Waiting for last bag confirmation"""
        if has_bag:
            # We have another bag - continue tracking
            self._change_state(AgentState.TRACKING_BAGS)
            self.context.frames_without_bag = 0
        else:
            self.context.frames_without_bag += 1
            
            # Wait for stable no-bag period
            stable_frames = self.config.get('stable_frames', 10)
            if self.context.frames_without_bag >= stable_frames:
                # Set last bag and complete
                if self.context.all_detections:
                    self.context.last_bag = self.context.all_detections[-1]
                    
                    # Process last bag OCR if not already done
                    last_bag_ocr = next((ocr for ocr in self.context.ocr_results 
                                       if ocr.bag_id == self.context.last_bag.bag_id), None)
                    
                    if not last_bag_ocr:
                        self._process_bag_ocr(self.context.last_bag, is_last=True)
                    else:
                        self.context.last_barcode = last_bag_ocr.barcode
                        self._change_state(AgentState.COMPLETED)
    
    def _process_bag_ocr(self, bag_detection: BagDetection, is_first: bool = False, is_last: bool = False):
        """Process OCR for a detected bag"""
        if self.current_frame is None:
            return
        
        barcode = self.ocr.extract_barcode(self.current_frame, bag_detection.bbox)
        
        if barcode:
            ocr_result = OCRResult(
                barcode=barcode,
                confidence=0.9,  # Would come from OCR engine
                timestamp=datetime.now(),
                bag_id=bag_detection.bag_id,
                bbox=bag_detection.bbox,
                is_first_bag=is_first,
                is_last_bag=is_last
            )
            
            self.context.ocr_results.append(ocr_result)
            
            if is_first:
                self.context.first_barcode = barcode
            if is_last:
                self.context.last_barcode = barcode
                self._change_state(AgentState.COMPLETED)
            
            logger.info(f"OCR result: {barcode} for {bag_detection.bag_id}")
    
    def _change_state(self, new_state: AgentState):
        """Change agent state with logging"""
        old_state = self.state
        self.state = new_state
        self.state_history.append((old_state.name, new_state.name, datetime.now()))
        logger.info(f"State change: {old_state.name} -> {new_state.name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'state': self.state.name,
            'session_id': self.context.session_id,
            'total_bags': self.context.total_bags,
            'first_bag': self.context.first_bag.bag_id if self.context.first_bag else None,
            'last_bag': self.context.last_bag.bag_id if self.context.last_bag else None,
            'first_barcode': self.context.first_barcode,
            'last_barcode': self.context.last_barcode,
            'frames_processed': self.frame_count,
            'frames_without_bag': self.context.frames_without_bag,
            'ocr_results': len(self.context.ocr_results),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0
        }
    
    def reset(self):
        """Reset the agent to initial state"""
        self.state = AgentState.IDLE
        self.context = PipelineContext()
        self.current_frame = None
        self.frame_count = 0
        self.processing_times = []
        self.state_history = []
        self.last_detection_time = None
        logger.info("Agent reset")

# Visualization functions
def draw_pipeline_frame(frame: np.ndarray, agent: BagDetectionAgent) -> np.ndarray:
    """Draw pipeline information on frame"""
    display_frame = frame.copy()
    status = agent.get_status()
    
    # Draw detection boxes
    if agent.context.all_detections:
        for detection in agent.context.all_detections[-5:]:  # Last 5 detections
            x1, y1, x2, y2 = detection.bbox
            
            # Color code: green for first, red for last, blue for others
            if detection.bag_id == status['first_bag']:
                color = (0, 255, 0)  # Green
            elif detection.bag_id == status['last_bag']:
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 0)  # Blue
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.bag_id}: {detection.confidence:.2f}"
            cv2.putText(display_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw status overlay
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 250), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
    
    # Status text
    state_colors = {
        'WAITING_FIRST_BAG': (0, 255, 255),
        'FIRST_BAG_DETECTED': (0, 255, 0),
        'TRACKING_BAGS': (255, 255, 0),
        'WAITING_LAST_BAG': (255, 165, 0),
        'COMPLETED': (0, 255, 0),
        'ERROR': (0, 0, 255)
    }
    
    color = state_colors.get(status['state'], (255, 255, 255))
    
    lines = [
        f"State: {status['state']}",
        f"Bags: {status['total_bags']}",
        f"First: {status['first_bag'] or 'None'}",
        f"Last: {status['last_bag'] or 'None'}",
        f"First Barcode: {status['first_barcode'] or 'None'}",
        f"Last Barcode: {status['last_barcode'] or 'None'}",
        f"Frames: {status['frames_processed']}",
        f"OCR Results: {status['ocr_results']}",
        f"Avg Time: {status['avg_processing_time']*1000:.1f}ms"
    ]
    
    for i, line in enumerate(lines):
        y_pos = 40 + i * 25
        cv2.putText(display_frame, line, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return display_frame

# Streamlit App
def main():
    st.title("🛍️ Bag Detection Pipeline Agent")
    st.markdown("**Stapipy + YOLO + OCR Pipeline** - Manages 1st bag detection, barcode extraction, and last bag detection")
    
    # Initialize agent
    if 'agent' not in st.session_state:
        config = {
            'confidence_threshold': 0.5,
            'last_bag_timeout': 3.0,
            'stable_frames': 5,
            'source_type': 'synthetic'
        }
        st.session_state.agent = BagDetectionAgent(config)
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Sidebar Configuration
    st.sidebar.header("Pipeline Configuration")
    
    st.sidebar.subheader("Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    last_bag_timeout = st.sidebar.slider("Last Bag Timeout (sec)", 1, 10, 3)
    
    st.sidebar.subheader("OCR Configuration")
    ocr_engine = st.sidebar.selectbox("OCR Engine", ["paddle", "simulated"])
    
    # Update agent config
    st.session_state.agent.config.update({
        'confidence_threshold': confidence_threshold,
        'last_bag_timeout': last_bag_timeout,
        'ocr_engine': ocr_engine
    })
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Pipeline Live View")
        
        # Controls
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            if st.button("🚀 Start Pipeline", use_container_width=True) and not st.session_state.processing:
                st.session_state.processing = True
                st.session_state.agent.start_pipeline()
        with col1b:
            if st.button("⏹️ Stop Pipeline", use_container_width=True):
                st.session_state.processing = False
                st.session_state.agent.stop_pipeline()
        with col1c:
            if st.button("🔄 Reset Pipeline", use_container_width=True):
                st.session_state.processing = False
                st.session_state.agent.reset()
                st.rerun()
        
        # Live feed
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Processing loop
        if st.session_state.processing:
            continue_processing = st.session_state.agent.process_single_frame()
            
            if st.session_state.agent.current_frame is not None:
                # Draw visualization
                display_frame = draw_pipeline_frame(
                    st.session_state.agent.current_frame, 
                    st.session_state.agent
                )
                
                # Convert to RGB for display
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_frame_rgb, caption="Live Pipeline View", use_column_width=True)
            
            # Update status
            status = st.session_state.agent.get_status()
            display_pipeline_status(status, status_placeholder)
            
            # Check if completed
            if not continue_processing:
                st.session_state.processing = False
                if status['state'] == 'COMPLETED':
                    st.balloons()
                    st.success("🎉 Pipeline completed successfully!")
                elif status['state'] == 'ERROR':
                    st.error("❌ Pipeline encountered an error")
        
        else:
            # Show static state when not processing
            if st.session_state.agent.current_frame is not None:
                display_frame = draw_pipeline_frame(
                    st.session_state.agent.current_frame, 
                    st.session_state.agent
                )
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_frame_rgb, caption="Pipeline Ready", use_column_width=True)
            else:
                # Generate initial frame
                frame_data = st.session_state.agent.stapipy.get_frame()
                if frame_data:
                    st.session_state.agent.current_frame = frame_data['frame']
                    display_frame = draw_pipeline_frame(
                        st.session_state.agent.current_frame, 
                        st.session_state.agent
                    )
                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame_rgb, caption="Pipeline Ready", use_column_width=True)
            
            status_placeholder.info("Click 'Start Pipeline' to begin processing")
    
    with col2:
        st.subheader("Pipeline Statistics")
        display_pipeline_statistics(st.session_state.agent)
    
    # Results section
    st.markdown("---")
    st.subheader("Pipeline Results")
    display_pipeline_results(st.session_state.agent.context)

def display_pipeline_status(status: Dict, placeholder):
    """Display current pipeline status"""
    with placeholder.container():
        st.subheader("Pipeline Status")
        
        state_icons = {
            'WAITING_FIRST_BAG': '⏳',
            'FIRST_BAG_DETECTED': '✅',
            'TRACKING_BAGS': '📦',
            'WAITING_LAST_BAG': '⏳',
            'COMPLETED': '🎉',
            'ERROR': '❌'
        }
        
        icon = state_icons.get(status['state'], '🔧')
        st.markdown(f"### {icon} {status['state']}")
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Bags", status['total_bags'])
            st.metric("Frames Processed", status['frames_processed'])
        with col2:
            st.metric("OCR Results", status['ocr_results'])
            st.metric("Avg Processing Time", f"{status['avg_processing_time']*1000:.1f}ms")
        
        # First/Last bag info
        if status['first_bag']:
            st.info(f"**First Bag:** {status['first_bag']} - Barcode: {status['first_barcode'] or 'Processing...'}")
        if status['last_bag']:
            st.info(f"**Last Bag:** {status['last_bag']} - Barcode: {status['last_barcode'] or 'Processing...'}")

def display_pipeline_statistics(agent: BagDetectionAgent):
    """Display pipeline statistics and charts"""
    if agent.frame_count > 0:
        status = agent.get_status()
        
        # State history
        if agent.state_history:
            df_states = pd.DataFrame(agent.state_history, columns=['old_state', 'new_state', 'timestamp'])
            df_states['time'] = pd.to_datetime(df_states['timestamp'])
            
            fig = px.timeline(df_states, x_start='time', y='new_state', color='new_state',
                             title='State Transition Timeline', height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.subheader("Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Session ID", status['session_id'][-8:])
        with col2:
            st.metric("State Changes", len(agent.state_history))
        with col3:
            st.metric("No Bag Frames", status['frames_without_bag'])
        
        # Processing time distribution
        if agent.processing_times:
            fig = px.histogram(x=agent.processing_times, 
                             title='Processing Time Distribution',
                             labels={'x': 'Time (seconds)', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Start the pipeline to see statistics")

def display_pipeline_results(context: PipelineContext):
    """Display pipeline results"""
    if context.ocr_results:
        st.success(f"Pipeline completed with {len(context.ocr_results)} barcode extractions")
        
        # Results table
        results_data = []
        for ocr in context.ocr_results:
            results_data.append({
                'Bag ID': ocr.bag_id,
                'Barcode': ocr.barcode,
                'Confidence': f"{ocr.confidence:.2f}",
                'First Bag': '✅' if ocr.is_first_bag else '',
                'Last Bag': '✅' if ocr.is_last_bag else '',
                'Timestamp': ocr.timestamp.strftime("%H:%M:%S")
            })
        
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Export results
        export_data = {
            'session_id': context.session_id,
            'pipeline_start': context.pipeline_start_time.isoformat() if context.pipeline_start_time else None,
            'first_bag': {
                'bag_id': context.first_bag.bag_id if context.first_bag else None,
                'barcode': context.first_barcode,
                'timestamp': context.first_bag.timestamp.isoformat() if context.first_bag else None
            },
            'last_bag': {
                'bag_id': context.last_bag.bag_id if context.last_bag else None,
                'barcode': context.last_barcode,
                'timestamp': context.last_bag.timestamp.isoformat() if context.last_bag else None
            },
            'total_bags': context.total_bags,
            'all_ocr_results': [
                {
                    'bag_id': ocr.bag_id,
                    'barcode': ocr.barcode,
                    'timestamp': ocr.timestamp.isoformat(),
                    'is_first': ocr.is_first_bag,
                    'is_last': ocr.is_last_bag
                } for ocr in context.ocr_results
            ]
        }
        
        json_data = json.dumps(export_data, indent=2)
        st.download_button(
            label="📥 Export Pipeline Results",
            data=json_data,
            file_name=f"pipeline_results_{context.session_id}.json",
            mime="application/json"
        )
    
    elif st.session_state.processing:
        st.info("Pipeline running... results will appear here when complete")
    else:
        st.warning("No pipeline results yet. Start the pipeline to begin processing.")

if __name__ == "__main__":
    main()
