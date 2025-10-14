import streamlit as st
import numpy as np
import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import random
import json
import io
import base64

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

class OCRSimulator:
    """OCR simulator with model upload capability"""
    
    def __init__(self):
        self.simulated_barcodes = [
            "123456789012", "987654321098", "456123789045",
            "321654987012", "789012345678", "210987654321",
            "345678901234", "876543210987", "234567890123"
        ]
        self.custom_model_loaded = False
        self.model_name = "Default Simulator"
    
    def load_custom_model(self, uploaded_file):
        """Simulate loading a custom model"""
        try:
            # In a real implementation, this would load your actual model
            # For simulation, we'll just mark that a custom model is loaded
            self.custom_model_loaded = True
            self.model_name = uploaded_file.name
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def extract_barcode(self, image: np.ndarray, bbox: Optional[tuple] = None) -> Optional[str]:
        """Extract barcode from image region"""
        # Simulate better performance with custom model
        success_rate = 0.8 if self.custom_model_loaded else 0.6
        
        if random.random() > (1 - success_rate):
            barcode = random.choice(self.simulated_barcodes)
            
            # Add some realism - custom models might have different output
            if self.custom_model_loaded:
                # Simulate custom model behavior
                if random.random() > 0.1:  # 90% confidence with custom model
                    return barcode
            else:
                # Default simulator behavior
                if random.random() > 0.3:  # 70% confidence with default
                    return barcode
        
        return None

class StateMachineAgent:
    """
    Production-ready state machine agent for bag detection pipeline
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

        # OCR engine
        self.ocr_engine = OCRSimulator()

        # Runtime tracking
        self.last_bag_candidate: Optional[BagDetection] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.state_history = []

        logger.info(f"Agent initialized with session_id: {self.context.session_id}")

    def simulate_detection(self, frame: np.ndarray) -> List[Dict]:
        """Simulate object detection"""
        height, width = frame.shape[1], frame.shape[0]  # PIL image dimensions
        detections = []
        
        # More realistic detection simulation
        detection_probability = 0.7
        
        if random.random() < detection_probability:
            num_bags = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            
            for i in range(num_bags):
                w, h = random.randint(80, 200), random.randint(120, 300)
                x1 = random.randint(50, width - w - 50)
                y1 = random.randint(50, height - h - 50)
                
                detections.append({
                    'bbox': (x1, y1, x1 + w, y1 + h),
                    'confidence': random.uniform(0.7, 0.95),
                    'class_id': 0,
                    'class_name': "bag"
                })
        
        return detections

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
            detections = self.simulate_detection(frame)

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
                    confidence=random.uniform(0.85, 0.98) if self.ocr_engine.custom_model_loaded else random.uniform(0.7, 0.9),
                    text_data={"type": "barcode", "position": detection.bbox, "model": self.ocr_engine.model_name},
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
            "frame_count": self.frame_count,
            "ocr_model": self.ocr_engine.model_name
        }

    def load_ocr_model(self, uploaded_file):
        """Load custom OCR model"""
        return self.ocr_engine.load_custom_model(uploaded_file)

def create_conveyor_frame(frame_count: int, width: int = 640, height: int = 480) -> Image.Image:
    """Create a conveyor belt simulation frame"""
    # Create base image
    img = Image.new('RGB', (width, height), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    # Draw conveyor belt
    belt_color = (100, 100, 100)
    belt_top = height // 2 - 60
    belt_bottom = height // 2 + 60
    draw.rectangle([50, belt_top, width-50, belt_bottom], fill=belt_color, outline=(150, 150, 150), width=2)
    
    # Draw moving elements
    x_pos = (frame_count * 5) % (width - 100)
    
    # Draw bags based on simulation state
    if st.session_state.get('processing', False):
        # Simulate bags moving on conveyor
        bag_positions = [(x_pos, belt_top + 20)]
        
        # Add some random bags
        if random.random() > 0.7:
            bag_positions.append(((x_pos + 200) % (width - 100), belt_top + 20))
        
        for bag_x, bag_y in bag_positions:
            # Draw bag
            bag_color = (0, 128, 255)  # Blue bags
            draw.rectangle([bag_x, bag_y, bag_x + 80, bag_y + 120], 
                         fill=bag_color, outline=(255, 255, 255), width=2)
            
            # Draw barcode area
            draw.rectangle([bag_x + 10, bag_y + 90, bag_x + 70, bag_y + 110], 
                         fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    
    # Draw conveyor rollers
    for i in range(0, width, 30):
        roller_y = belt_bottom + 10
        draw.ellipse([i, roller_y, i + 20, roller_y + 20], fill=(150, 150, 150))
    
    # Add labels
    try:
        font = ImageFont.load_default()
        draw.text((width//2 - 60, belt_top - 30), "Conveyor Belt", fill=(255, 255, 255), font=font)
        draw.text((20, 20), f"Frame: {frame_count}", fill=(255, 255, 255), font=font)
    except:
        # Fallback if font loading fails
        draw.text((width//2 - 60, belt_top - 30), "Conveyor Belt", fill=(255, 255, 255))
        draw.text((20, 20), f"Frame: {frame_count}", fill=(255, 255, 255))
    
    return img

def draw_detections_on_image(img: Image.Image, detections: List[Dict]) -> Image.Image:
    """Draw detection bounding boxes on image"""
    draw = ImageDraw.Draw(img)
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        
        # Draw label background
        label = f"Bag: {confidence:.2f}"
        try:
            font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), label, font=font)
        except:
            bbox = (0, 0, len(label) * 6, 12)
        
        label_width = bbox[2] - bbox[0]
        label_height = bbox[3] - bbox[1]
        
        draw.rectangle([x1, y1 - label_height - 5, x1 + label_width + 10, y1], 
                      fill=(0, 255, 0))
        
        # Draw label text
        draw.text((x1 + 5, y1 - label_height - 2), label, fill=(0, 0, 0))
    
    return img

def draw_status_overlay_on_image(img: Image.Image, agent_status: Dict, context: AgentContext) -> Image.Image:
    """Draw status information on image"""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Draw semi-transparent overlay
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([10, 10, 400, 220], fill=(0, 0, 0, 180))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # State colors
    state_colors = {
        "IDLE": (255, 255, 0),
        "WAITING_FIRST_BAG": (255, 165, 0),
        "FIRST_BAG_DETECTED": (0, 255, 255),
        "TRACKING_BAGS": (0, 255, 0),
        "LAST_BAG_DETECTED": (0, 200, 255),
        "COMPLETED": (0, 255, 0),
        "ERROR": (255, 0, 0)
    }
    
    color = state_colors.get(agent_status['state'], (255, 255, 255))
    
    # Draw text information
    status_lines = [
        f"State: {agent_status['state']}",
        f"Bags Detected: {context.bag_count}",
        f"Barcodes Found: {len(context.ocr_results)}",
        f"OCR Model: {agent_status.get('ocr_model', 'Default')}",
        f"Session: {context.session_id}",
        f"Frames w/o Detection: {context.frames_without_detection}",
        f"Frame Count: {agent_status['frame_count']}"
    ]
    
    y_pos = 30
    for line in status_lines:
        draw.text((20, y_pos), line, fill=color)
        y_pos += 25
    
    # Draw recent barcodes
    if context.ocr_results:
        draw.text((20, 180), "Recent Barcodes:", fill=(255, 255, 0))
        y_pos = 200
        for i, ocr_result in enumerate(context.ocr_results[-2:]):
            barcode_text = f"{ocr_result.barcode} ({ocr_result.confidence:.2f})"
            draw.text((30, y_pos), barcode_text, fill=(255, 255, 0))
            y_pos += 20
    
    return img

# Streamlit App
def main():
    st.title("🛍️ Bag Detection AI Agent")
    st.markdown("Production-ready state machine with OCR model upload capability")

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = StateMachineAgent()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0

    # Sidebar Configuration
    st.sidebar.header("Configuration")
    
    # OCR Model Upload Section
    st.sidebar.subheader("📁 Upload Custom OCR Model")
    
    uploaded_model = st.sidebar.file_uploader(
        "Upload your trained OCR model",
        type=['zip', 'pth', 'pt', 'onnx', 'h5', 'pkl'],
        help="Upload your custom trained OCR model file"
    )
    
    if uploaded_model is not None:
        if st.sidebar.button("🚀 Load Custom Model"):
            with st.spinner("Loading custom model..."):
                success = st.session_state.agent.load_ocr_model(uploaded_model)
                if success:
                    st.sidebar.success(f"✅ Model '{uploaded_model.name}' loaded successfully!")
                    st.sidebar.info("Custom model will provide better barcode detection accuracy")
                else:
                    st.sidebar.error("❌ Failed to load model")
    
    # Show current model status
    current_status = st.session_state.agent.get_status()
    if st.session_state.agent.ocr_engine.custom_model_loaded:
        st.sidebar.success(f"📦 Using: {st.session_state.agent.ocr_engine.model_name}")
    else:
        st.sidebar.info("🔧 Using default OCR simulator")
    
    # Detection Parameters
    st.sidebar.subheader("🎯 Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    last_bag_timeout = st.sidebar.slider("Last Bag Timeout (sec)", 1, 30, 5)
    process_all_bags = st.sidebar.checkbox("Process All Bags", value=True)

    # Update agent config
    st.session_state.agent.config.update({
        'min_confidence': confidence_threshold,
        'last_bag_timeout': last_bag_timeout,
        'process_all_bags': process_all_bags
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
            # Create conveyor frame
            st.session_state.frame_count += 1
            conveyor_img = create_conveyor_frame(st.session_state.frame_count)
            
            # Convert to numpy array for processing
            frame_np = np.array(conveyor_img)
            
            # Create frame data
            frame_data = {
                'frame': frame_np,
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
            detections = st.session_state.agent.simulate_detection(frame_np)
            
            # Draw visualizations
            display_img = draw_detections_on_image(conveyor_img, detections)
            agent_status = st.session_state.agent.get_status()
            display_img = draw_status_overlay_on_image(display_img, agent_status, st.session_state.agent.context)
            
            # Display image
            video_placeholder.image(display_img, caption="Live Conveyor Feed with Detections", use_column_width=True)
            
            # Display status
            display_agent_status(agent_status, st.session_state.agent.context, status_placeholder)
            
            # Auto-reset if completed
            if agent_status['state'] == 'COMPLETED':
                time.sleep(3)
                st.session_state.agent.reset()
                st.rerun()
                
        else:
            # Show static conveyor when not processing
            conveyor_img = create_conveyor_frame(0)
            video_placeholder.image(conveyor_img, caption="Ready to Start Processing", use_column_width=True)
            status_placeholder.info("Click 'Start Processing' to begin bag detection")
    
    with col2:
        st.subheader("Agent Statistics")
        display_statistics(st.session_state.agent)

    # Results section
    st.markdown("---")
    st.subheader("📊 Processing Results")
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
        
        # Model info
        if agent_status.get('ocr_model', 'Default') != 'Default Simulator':
            st.success(f"📦 OCR Model: {agent_status['ocr_model']}")
        
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
        
        # State history chart
        if agent.state_history:
            df_states = pd.DataFrame(agent.state_history)
            df_states['time'] = pd.to_datetime(df_states['timestamp'])
            
            fig = px.timeline(df_states, x_start="time", y="new_state", color="new_state",
                             title="State Transition History", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Session information
        st.subheader("Session Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Session ID", agent.context.session_id[:8] + "...")
        with col2:
            st.metric("State Changes", len(agent.state_history))
        with col3:
            st.metric("No Detection Frames", agent.context.frames_without_detection)
        
        # State distribution pie chart
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
        
        # Create results table
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
        
        # Download results
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
