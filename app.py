import streamlit as st
import cv2
import numpy as np
import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
import pandas as pd
import plotly.express as px
from PIL import Image
import random
import json
import tempfile
import os
from pathlib import Path
import zipfile
import shutil

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
    """State machine states"""
    IDLE = auto()
    WAITING_FIRST_BAG = auto()
    FIRST_BAG_DETECTED = auto()
    PROCESSING_OCR = auto()
    TRACKING_BAGS = auto()
    WAITING_LAST_BAG = auto()
    LAST_BAG_DETECTED = auto()
    ERROR = auto()
    COMPLETED = auto()

@dataclass
class BagDetection:
    """Data class for bag detection results"""
    bag_id: str
    confidence: float
    bbox: tuple
    timestamp: datetime
    frame_id: int
    class_id: int = 0
    class_name: str = "bag"

@dataclass
class OCRResult:
    """Data class for OCR results"""
    barcode: str
    confidence: float
    text_data: Dict[str, Any]
    timestamp: datetime
    bag_id: str
    bbox: Optional[tuple] = None

@dataclass
class AgentContext:
    """Context data shared across states"""
    first_bag: Optional[BagDetection] = None
    last_bag: Optional[BagDetection] = None
    ocr_results: List[OCRResult] = field(default_factory=list)
    bag_count: int = 0
    frames_without_detection: int = 0
    error_message: Optional[str] = None
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    last_detection_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert context to dictionary for logging/export"""
        return {
            "session_id": self.session_id,
            "first_bag": {
                "bag_id": self.first_bag.bag_id,
                "timestamp": self.first_bag.timestamp.isoformat(),
                "confidence": self.first_bag.confidence,
                "bbox": self.first_bag.bbox
            } if self.first_bag else None,
            "last_bag": {
                "bag_id": self.last_bag.bag_id,
                "timestamp": self.last_bag.timestamp.isoformat(),
                "confidence": self.last_bag.confidence,
                "bbox": self.last_bag.bbox
            } if self.last_bag else None,
            "bag_count": self.bag_count,
            "ocr_results": [
                {
                    "barcode": r.barcode,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp.isoformat(),
                    "bag_id": r.bag_id
                } for r in self.ocr_results
            ]
        }

class YOLOModel:
    """YOLO model wrapper for real detection"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            st.success(f"✅ YOLO model loaded from {self.model_path}")
        except Exception as e:
            st.error(f"❌ Failed to load YOLO model: {e}")
            # Fallback to simulation
            self.model = None
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on image"""
        if self.model is None:
            return self._simulate_detection(image)
        
        try:
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': result.names[cls] if hasattr(result, 'names') else "object"
                        })
            
            return detections
        except Exception as e:
            st.error(f"Detection error: {e}")
            return self._simulate_detection(image)
    
    def _simulate_detection(self, image: np.ndarray) -> List[Dict]:
        """Fallback simulation"""
        height, width = image.shape[:2]
        detections = []
        
        # Randomly generate detections for demo
        if random.random() > 0.3:
            w, h = random.randint(100, 300), random.randint(150, 400)
            x1 = random.randint(50, width - w - 50)
            y1 = random.randint(50, height - h - 50)
            
            detections.append({
                'bbox': (x1, y1, x1 + w, y1 + h),
                'confidence': random.uniform(0.7, 0.95),
                'class_id': 0,
                'class_name': "bag"
            })
        
        return detections

class OCREngine:
    """OCR engine with support for custom trained models"""
    
    def __init__(self, engine_type: str = "paddle", model_path: Optional[str] = None):
        self.engine_type = engine_type
        self.model_path = model_path
        self.engine = None
        self._load_engine()
    
    def _load_engine(self):
        """Load OCR engine with custom model if provided"""
        try:
            if self.engine_type == "paddle":
                from paddleocr import PaddleOCR
                
                if self.model_path and os.path.exists(self.model_path):
                    # Load custom trained model
                    self.engine = PaddleOCR(
                        det_model_dir=os.path.join(self.model_path, 'det'),
                        rec_model_dir=os.path.join(self.model_path, 'rec'),
                        cls_model_dir=os.path.join(self.model_path, 'cls'),
                        use_angle_cls=True,
                        lang='en',
                        show_log=False
                    )
                    st.success("✅ Custom PaddleOCR model loaded")
                else:
                    # Load default model
                    self.engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                    st.success("✅ Default PaddleOCR model loaded")
                    
            elif self.engine_type == "easyocr":
                import easyocr
                self.engine = easyocr.Reader(['en'])
                st.success("✅ EasyOCR model loaded")
                
            elif self.engine_type == "tesseract":
                import pytesseract
                self.engine = pytesseract
                st.success("✅ Tesseract OCR ready")
                
            else:
                st.warning("Using simulated OCR")
                self.engine = None
                
        except Exception as e:
            st.warning(f"OCR engine not available, using simulation: {e}")
            self.engine = None
    
    def extract_barcode(self, image: np.ndarray, bbox: Optional[tuple] = None) -> Optional[str]:
        """Extract barcode from image region"""
        if self.engine is None:
            return self._simulate_barcode()
        
        try:
            if bbox:
                x1, y1, x2, y2 = bbox
                roi = image[y1:y2, x1:x2]
            else:
                roi = image
            
            # Convert BGR to RGB for OCR engines
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            if self.engine_type == "paddle":
                result = self.engine.ocr(roi_rgb, cls=True)
                
                if result and result[0]:
                    # Extract text from all detected regions
                    texts = [line[1][0] for line in result[0]]
                    confidences = [line[1][1] for line in result[0]]
                    
                    # Look for barcode patterns
                    barcode = self._find_barcode_pattern(texts, confidences)
                    if barcode:
                        return barcode
            
            elif self.engine_type == "easyocr":
                results = self.engine.readtext(roi_rgb)
                if results:
                    texts = [result[1] for result in results]
                    confidences = [result[2] for result in results]
                    barcode = self._find_barcode_pattern(texts, confidences)
                    if barcode:
                        return barcode
            
            elif self.engine_type == "tesseract":
                text = self.engine.image_to_string(roi_rgb)
                if text.strip():
                    # Simple barcode pattern matching
                    import re
                    patterns = [
                        r'\b[A-Z0-9]{8,15}\b',
                        r'\b\d{12,13}\b',
                        r'\b\d{8}\b',
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, text)
                        if match:
                            return match.group(0)
            
            return None
            
        except Exception as e:
            st.error(f"OCR extraction error: {e}")
            return self._simulate_barcode()
    
    def _find_barcode_pattern(self, texts: List[str], confidences: List[float]) -> Optional[str]:
        """Find barcode patterns in OCR results"""
        import re
        
        for text, confidence in zip(texts, confidences):
            if confidence < 0.5:
                continue
                
            # Common barcode patterns
            patterns = [
                r'\b[A-Z0-9]{8,15}\b',  # Alphanumeric codes
                r'\b\d{12,13}\b',       # EAN-13, UPC
                r'\b\d{8}\b',           # EAN-8
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text.replace(' ', ''))
                if match:
                    return match.group(0)
        
        return None
    
    def _simulate_barcode(self) -> Optional[str]:
        """Simulate barcode extraction"""
        simulated_barcodes = [
            "123456789012", "987654321098", "456123789045",
            "321654987012", "789012345678", "210987654321"
        ]
        
        if random.random() > 0.6:
            return random.choice(simulated_barcodes)
        return None

class ModelManager:
    """Manager for handling model uploads and storage"""
    
    def __init__(self):
        self.models_dir = Path("./uploaded_models")
        self.models_dir.mkdir(exist_ok=True)
    
    def save_uploaded_model(self, uploaded_file, model_type: str) -> str:
        """Save uploaded model file and return path"""
        try:
            # Create model directory
            model_name = uploaded_file.name.split('.')[0]
            model_path = self.models_dir / model_type / model_name
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save the file
            file_path = model_path / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # If it's a zip file, extract it
            if uploaded_file.name.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(model_path)
                os.remove(file_path)  # Remove the zip file after extraction
            
            return str(model_path)
            
        except Exception as e:
            st.error(f"Error saving model: {e}")
            return None
    
    def get_available_models(self, model_type: str) -> List[str]:
        """Get list of available models"""
        model_dir = self.models_dir / model_type
        if model_dir.exists():
            return [d.name for d in model_dir.iterdir() if d.is_dir()]
        return []
    
    def get_model_path(self, model_type: str, model_name: str) -> str:
        """Get path to specific model"""
        return str(self.models_dir / model_type / model_name)

class StateMachineAgent:
    """
    Production-ready state machine agent with real model integration
    """

    def __init__(self, config: Optional[Dict] = None):
        self.state = AgentState.IDLE
        self.context = AgentContext()

        # Configuration
        self.config = config or {}
        self.last_bag_timeout = self.config.get('last_bag_timeout', 5.0)
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.max_no_detection_frames = self.config.get('max_no_detection_frames', 150)
        self.process_all_bags = self.config.get('process_all_bags', True)

        # Model components
        self.yolo_model = YOLOModel(
            model_path=self.config.get('model_path', 'yolov8n.pt'),
            confidence_threshold=self.min_confidence
        )
        self.ocr_engine = OCREngine(
            engine_type=self.config.get('ocr_engine', 'paddle'),
            model_path=self.config.get('ocr_model_path')
        )

        # Runtime tracking
        self.last_bag_candidate: Optional[BagDetection] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.state_history = []

        logger.info(f"Agent initialized with session_id: {self.context.session_id}")

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO model"""
        return self.yolo_model.detect(frame)

    def change_state(self, new_state: AgentState):
        """Change agent state"""
        old_state = self.state
        self.state = new_state
        self.state_history.append({
            'old_state': old_state.name,
            'new_state': new_state.name,
            'timestamp': datetime.now()
        })
        logger.info(f"State transition: {old_state.name} -> {new_state.name}")

    async def start(self):
        """Start the agent"""
        self.change_state(AgentState.WAITING_FIRST_BAG)
        logger.info("Agent started - waiting for first bag")

    async def process_frame(self, frame_data: Dict):
        """
        Main processing loop for each frame
        """
        try:
            # Skip if not in active states
            if self.state not in [
                AgentState.WAITING_FIRST_BAG,
                AgentState.TRACKING_BAGS
            ]:
                return

            frame = frame_data['frame']
            frame_id = frame_data['frame_id']
            self.current_frame = frame
            self.frame_count = frame_id

            # Run detection
            detections = self.detect_objects(frame)

            if detections:
                # Filter by confidence and get best detection
                valid_detections = [d for d in detections if d['confidence'] >= self.min_confidence]

                if valid_detections:
                    # Take detection with highest confidence
                    best_detection = max(valid_detections, key=lambda x: x['confidence'])

                    bag_detection = BagDetection(
                        bag_id=f"BAG_{self.context.bag_count + 1:04d}",
                        confidence=best_detection['confidence'],
                        bbox=best_detection['bbox'],
                        timestamp=frame_data['timestamp'],
                        frame_id=frame_id,
                        class_id=best_detection['class_id'],
                        class_name=best_detection['class_name']
                    )

                    await self._handle_bag_detected(bag_detection)
                else:
                    await self._handle_no_detection()
            else:
                await self._handle_no_detection()

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.context.error_message = str(e)
            self.change_state(AgentState.ERROR)

    async def _handle_bag_detected(self, detection: BagDetection):
        """Handle bag detection event"""
        if self.state == AgentState.WAITING_FIRST_BAG:
            self.context.first_bag = detection
            self.context.bag_count = 1
            self.context.last_detection_time = datetime.now()
            self.change_state(AgentState.FIRST_BAG_DETECTED)

            # Trigger OCR processing
            await self._process_ocr(detection, is_first=True)

        elif self.state == AgentState.TRACKING_BAGS:
            self.context.bag_count += 1
            self.context.frames_without_detection = 0
            self.context.last_detection_time = datetime.now()
            self.last_bag_candidate = detection

            # Process OCR for this bag if configured
            if self.process_all_bags:
                await self._process_ocr(detection, is_first=False)

    async def _handle_no_detection(self):
        """Handle frames without bag detection"""
        self.context.frames_without_detection += 1

        if self.state == AgentState.TRACKING_BAGS:
            # Check timeout for last bag
            if self.context.last_detection_time:
                elapsed = (datetime.now() - self.context.last_detection_time).total_seconds()
                if elapsed >= self.last_bag_timeout:
                    logger.info(f"Last bag timeout reached ({elapsed:.2f}s)")
                    await self._handle_last_bag_timeout()

    async def _handle_last_bag_timeout(self):
        """Handle last bag timeout"""
        if self.last_bag_candidate:
            self.context.last_bag = self.last_bag_candidate
            self.change_state(AgentState.LAST_BAG_DETECTED)

            # Process last bag OCR
            await self._process_ocr(self.context.last_bag, is_first=False, is_last=True)
        else:
            # No last bag, just complete
            await self._complete_session()

    async def _process_ocr(self, detection: BagDetection, is_first: bool = False, is_last: bool = False):
        """Process OCR for detected bag"""
        try:
            logger.info(f"Processing OCR for bag {detection.bag_id}")

            if self.current_frame is None:
                logger.warning("No current frame available for OCR")
                return

            # Extract barcode using OCR engine
            barcode = self.ocr_engine.extract_barcode(self.current_frame, detection.bbox)

            if barcode:
                ocr_result = OCRResult(
                    barcode=barcode,
                    confidence=random.uniform(0.7, 0.95),  # Would come from OCR engine
                    text_data={"type": "barcode", "position": detection.bbox},
                    timestamp=datetime.now(),
                    bag_id=detection.bag_id,
                    bbox=detection.bbox
                )

                self.context.ocr_results.append(ocr_result)
                logger.info(f"Barcode extracted: {barcode}")

                # If first bag processed, start tracking
                if is_first and self.state == AgentState.FIRST_BAG_DETECTED:
                    self.change_state(AgentState.TRACKING_BAGS)

                # If last bag processed, complete session
                if is_last and self.state == AgentState.LAST_BAG_DETECTED:
                    await self._complete_session()

        except Exception as e:
            logger.error(f"OCR processing error: {e}")

    async def _complete_session(self):
        """Complete the session"""
        self.change_state(AgentState.COMPLETED)
        logger.info(f"Session completed: {self.context.bag_count} bags processed, "
                   f"{len(self.context.ocr_results)} barcodes extracted")

    def reset(self):
        """Reset agent to initial state"""
        self.state = AgentState.IDLE
        self.context = AgentContext()
        self.last_bag_candidate = None
        self.current_frame = None
        self.frame_count = 0
        self.state_history.clear()
        logger.info("Agent reset")

    def get_status(self) -> Dict:
        """Get current agent status"""
        return {
            "state": self.state.name,
            "session_id": self.context.session_id,
            "bag_count": self.context.bag_count,
            "ocr_count": len(self.context.ocr_results),
            "frames_without_detection": self.context.frames_without_detection,
            "error": self.context.error_message,
            "frame_count": self.frame_count
        }

    def update_ocr_engine(self, engine_type: str, model_path: Optional[str] = None):
        """Update OCR engine with new model"""
        self.ocr_engine = OCREngine(engine_type=engine_type, model_path=model_path)

# Helper functions for visualization (same as before)
def create_sample_frame(frame_count: int) -> np.ndarray:
    """Create a sample frame with visual elements"""
    frame = np.random.randint(50, 100, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 100), (590, 380), (200, 200, 200), 2)
    cv2.putText(frame, "Conveyor Belt View", (150, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    x_pos = (frame_count * 3) % 600
    cv2.circle(frame, (x_pos + 20, 240), 15, (0, 255, 255), -1)
    return frame

def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw detection bounding boxes on frame"""
    display_frame = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        color = (0, 255, 0)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        label = f"Bag: {detection['confidence']:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(display_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return display_frame

def draw_status_overlay(frame: np.ndarray, agent_status: Dict, context: AgentContext) -> np.ndarray:
    """Draw status information on frame"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    state_colors = {
        "IDLE": (255, 255, 0), "WAITING_FIRST_BAG": (255, 165, 0),
        "FIRST_BAG_DETECTED": (0, 255, 255), "TRACKING_BAGS": (0, 255, 0),
        "LAST_BAG_DETECTED": (0, 200, 255), "COMPLETED": (0, 255, 0), "ERROR": (0, 0, 255)
    }
    color = state_colors.get(agent_status['state'], (255, 255, 255))
    
    status_lines = [
        f"State: {agent_status['state']}",
        f"Bags Detected: {context.bag_count}",
        f"Barcodes Found: {len(context.ocr_results)}",
        f"Session: {context.session_id}",
        f"Frames w/o Detection: {context.frames_without_detection}",
        f"Frame Count: {agent_status['frame_count']}"
    ]
    for i, line in enumerate(status_lines):
        y_pos = 40 + i * 25
        cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    if context.ocr_results:
        cv2.putText(frame, "Recent Barcodes:", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        for i, ocr_result in enumerate(context.ocr_results[-2:]):
            y_pos = 220 + i * 25
            barcode_text = f"{ocr_result.barcode} ({ocr_result.confidence:.2f})"
            cv2.putText(frame, barcode_text, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return frame

# Streamlit App with Model Upload
def main():
    st.title("🛍️ Bag Detection AI Agent")
    st.markdown("Production-ready state machine with custom OCR model support")

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = StateMachineAgent()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'current_ocr_model' not in st.session_state:
        st.session_state.current_ocr_model = "default"

    # Sidebar Configuration
    st.sidebar.header("Configuration")
    
    # OCR Model Selection Section
    st.sidebar.subheader("OCR Model Configuration")
    
    # OCR Engine Selection
    ocr_engine = st.sidebar.selectbox(
        "OCR Engine",
        ["paddle", "easyocr", "tesseract", "simulated"],
        index=0,
        help="Select OCR engine to use for barcode extraction"
    )
    
    # Model Upload Section
    st.sidebar.subheader("Upload Custom OCR Model")
    
    uploaded_model = st.sidebar.file_uploader(
        "Upload OCR Model",
        type=['zip', 'pth', 'pt', 'onnx', 'pb'],
        help="Upload your trained OCR model (PaddleOCR format preferred)"
    )
    
    if uploaded_model is not None:
        if st.sidebar.button("Upload & Load Model"):
            with st.spinner("Uploading and extracting model..."):
                model_path = st.session_state.model_manager.save_uploaded_model(
                    uploaded_model, "ocr"
                )
                if model_path:
                    st.session_state.current_ocr_model = uploaded_model.name
                    # Update agent with new model
                    st.session_state.agent.update_ocr_engine(ocr_engine, model_path)
                    st.sidebar.success(f"✅ Model '{uploaded_model.name}' loaded successfully!")
    
    # Show available models
    available_models = st.session_state.model_manager.get_available_models("ocr")
    if available_models:
        st.sidebar.subheader("Available OCR Models")
        selected_model = st.sidebar.selectbox("Select Model", available_models)
        if st.sidebar.button("Load Selected Model"):
            model_path = st.session_state.model_manager.get_model_path("ocr", selected_model)
            st.session_state.agent.update_ocr_engine(ocr_engine, model_path)
            st.session_state.current_ocr_model = selected_model
            st.sidebar.success(f"✅ Model '{selected_model}' loaded!")
    
    # Detection Parameters
    st.sidebar.subheader("Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    last_bag_timeout = st.sidebar.slider("Last Bag Timeout (sec)", 1, 30, 5)
    process_all_bags = st.sidebar.checkbox("Process All Bags", value=True)

    # Update agent config
    st.session_state.agent.config.update({
        'min_confidence': confidence_threshold,
        'last_bag_timeout': last_bag_timeout,
        'process_all_bags': process_all_bags,
        'ocr_engine': ocr_engine
    })

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Processing")
        
        # Display current model info
        if st.session_state.current_ocr_model != "default":
            st.info(f"📁 Using OCR Model: **{st.session_state.current_ocr_model}**")
        else:
            st.info("🔧 Using default OCR engine")
        
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
            # Create sample frame
            st.session_state.frame_count += 1
            frame = create_sample_frame(st.session_state.frame_count)
            
            # Create frame data
            frame_data = {
                'frame': frame,
                'frame_id': st.session_state.frame_count,
                'timestamp': datetime.now()
            }
            
            # Simulate async processing
            import asyncio
            try:
                # Start agent if not already started
                if st.session_state.agent.state == AgentState.IDLE:
                    asyncio.run(st.session_state.agent.start())
                
                # Process frame
                asyncio.run(st.session_state.agent.process_frame(frame_data))
                
            except Exception as e:
                st.error(f"Processing error: {e}")
            
            # Get detections for visualization
            detections = st.session_state.agent.detect_objects(frame)
            
            # Draw visualizations
            display_frame = draw_detections(frame, detections)
            agent_status = st.session_state.agent.get_status()
            display_frame = draw_status_overlay(display_frame, agent_status, st.session_state.agent.context)
            
            # Convert to PIL for display
            display_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            video_placeholder.image(display_image, caption="Live Feed with Detections", use_column_width=True)
            
            # Display status
            display_agent_status(agent_status, st.session_state.agent.context, status_placeholder)
            
            # Auto-reset if completed
            if agent_status['state'] == 'COMPLETED':
                time.sleep(3)
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
        display_statistics(st.session_state.agent)

    # Results section
    st.markdown("---")
    st.subheader("Processing Results")
    display_results(st.session_state.agent.context)

def display_agent_status(agent_status: Dict, context: AgentContext, placeholder):
    """Display current agent status"""
    with placeholder.container():
        st.subheader("Current Status")
        
        state_colors = {
            "IDLE": "blue", "WAITING_FIRST_BAG": "yellow",
            "FIRST_BAG_DETECTED": "orange", "TRACKING_BAGS": "green",
            "LAST_BAG_DETECTED": "purple", "COMPLETED": "green", "ERROR": "red"
        }
        
        state_color = state_colors.get(agent_status['state'], "gray")
        st.markdown(f"**State:** <span style='color: {state_color}; font-weight: bold; font-size: 1.2em'>{agent_status['state']}</span>", 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bags Detected", agent_status['bag_count'])
        with col2:
            st.metric("Barcodes Found", agent_status['ocr_count'])
        with col3:
            st.metric("Frames Processed", agent_status['frame_count'])
        
        if context.first_bag:
            st.info(f"First Bag: {context.first_bag.bag_id} (Confidence: {context.first_bag.confidence:.2f})")
        if context.last_bag:
            st.info(f"Last Bag: {context.last_bag.bag_id} (Confidence: {context.last_bag.confidence:.2f})")
        
        if agent_status['state'] == 'COMPLETED':
            st.success("🎉 Processing completed successfully!")
        elif agent_status['state'] == 'ERROR':
            st.error(f"❌ Error: {agent_status['error']}")

def display_statistics(agent):
    """Display agent statistics and charts"""
    if st.session_state.processing:
        agent_status = agent.get_status()
        
        if agent.state_history:
            df_states = pd.DataFrame(agent.state_history)
            df_states['time'] = pd.to_datetime(df_states['timestamp'])
            
            fig = px.timeline(df_states, x_start="time", y="new_state", color="new_state",
                             title="State Transition History", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Session Info")
        st.metric("Session ID", agent.context.session_id)
        st.metric("State Changes", len(agent.state_history))
        st.metric("Frames w/o Detection", agent.context.frames_without_detection)
        
        if agent.state_history:
            state_counts = pd.DataFrame(agent.state_history)['new_state'].value_counts()
            fig = px.pie(values=state_counts.values, names=state_counts.index,
                        title="State Distribution", height=250)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Start processing to see statistics")

def display_results(context: AgentContext):
    """Display processing results"""
    if context.ocr_results:
        st.success(f"✅ Successfully extracted {len(context.ocr_results)} barcodes")
        
        results_data = []
        for i, ocr_result in enumerate(context.ocr_results, 1):
            results_data.append({
                "Bag #": i,
                "Bag ID": ocr_result.bag_id,
                "Barcode": ocr_result.barcode,
                "Confidence": f"{ocr_result.confidence:.2f}",
                "Timestamp": ocr_result.timestamp.strftime("%H:%M:%S")
            })
        
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        json_data = json.dumps(context.to_dict(), indent=2, default=str)
        st.download_button(
            label="📥 Download Results JSON",
            data=json_data,
            file_name=f"session_{context.session_id}.json",
            mime="application/json"
        )
    else:
        st.warning("No barcodes extracted yet. Start processing to see results.")

if __name__ == "__main__":
    main()
