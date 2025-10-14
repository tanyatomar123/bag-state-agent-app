import streamlit as st
import numpy as np
import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import json
import random
import io
import base64

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
    """Bag detection result"""
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
    """Simulates Stapipy frame grabbing using PIL only"""
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.frame_count = 0
        self.bag_positions = []
    
    def get_frame(self) -> Dict:
        """Generate a conveyor belt frame"""
        self.frame_count += 1
        
        # Create base image
        img = Image.new('RGB', (self.width, self.height), color=(50, 50, 50))
        draw = ImageDraw.Draw(img)
        
        # Draw conveyor belt
        belt_top = self.height // 2 - 60
        belt_bottom = self.height // 2 + 60
        
        # Conveyor belt
        draw.rectangle([50, belt_top, self.width-50, belt_bottom], 
                      fill=(100, 100, 100), outline=(150, 150, 150), width=2)
        
        # Update bag positions (moving right to left)
        self._update_bag_positions()
        
        # Draw bags
        self.bag_positions = []
        num_bags = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
        
        for i in range(num_bags):
            bag_x = 50 + ((self.frame_count * 4 + i * 200) % (self.width - 200))
            bag_y = belt_top + 10
            bag_width, bag_height = 80, 120
            
            # Draw bag
            draw.rectangle([bag_x, bag_y, bag_x + bag_width, bag_y + bag_height], 
                          fill=(0, 100, 200), outline=(255, 255, 255), width=2)
            
            # Draw barcode area (white rectangle with black lines)
            barcode_y = bag_y + bag_height - 25
            draw.rectangle([bag_x + 10, barcode_y, bag_x + bag_width - 10, bag_y + bag_height - 5], 
                          fill=(255, 255, 255))
            
            # Draw barcode lines
            for line_x in range(bag_x + 15, bag_x + bag_width - 15, 4):
                if random.random() > 0.3:
                    line_height = random.randint(8, 15)
                    draw.rectangle([line_x, barcode_y, line_x + 2, barcode_y + line_height], 
                                  fill=(0, 0, 0))
            
            self.bag_positions.append({
                'bbox': (bag_x, bag_y, bag_x + bag_width, bag_y + bag_height),
                'confidence': random.uniform(0.7, 0.95)
            })
        
        # Draw conveyor text
        try:
            font = ImageFont.load_default()
            draw.text((self.width//2 - 70, belt_top - 30), "Conveyor Belt - Stapipy Stream", 
                     fill=(255, 255, 255), font=font)
            draw.text((20, 20), f"Frame: {self.frame_count}", 
                     fill=(255, 255, 255), font=font)
        except:
            draw.text((self.width//2 - 70, belt_top - 30), "Conveyor Belt - Stapipy Stream", 
                     fill=(255, 255, 255))
            draw.text((20, 20), f"Frame: {self.frame_count}", 
                     fill=(255, 255, 255))
        
        # Convert to numpy for processing
        frame_np = np.array(img)
        
        return {
            'frame': frame_np,
            'frame_id': self.frame_count,
            'timestamp': datetime.now(),
            'source': 'stapipy_simulator',
            'bag_positions': self.bag_positions
        }
    
    def _update_bag_positions(self):
        """Update bag positions for movement"""
        # This is handled in get_frame during drawing
        pass

class YOLOSimulator:
    """Simulates YOLO detection using the bag positions from Stapipy"""
    
    def __init__(self, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
    
    def detect(self, frame_data: Dict) -> List[Dict]:
        """Detect bags in frame - uses pre-computed positions from Stapipy"""
        detections = []
        
        # Use the bag positions from Stapipy simulator
        for bag in frame_data.get('bag_positions', []):
            if bag['confidence'] >= self.conf_threshold:
                detections.append({
                    'bbox': bag['bbox'],
                    'confidence': bag['confidence'],
                    'class_id': 0,
                    'class_name': 'bag'
                })
        
        # Occasionally add false positives or miss detections for realism
        if random.random() < 0.1:  # 10% chance of false positive
            false_positive = {
                'bbox': (
                    random.randint(50, 500),
                    random.randint(150, 250),
                    random.randint(100, 600),
                    random.randint(200, 350)
                ),
                'confidence': random.uniform(0.3, 0.6),  # Low confidence
                'class_id': 0,
                'class_name': 'bag'
            }
            detections.append(false_positive)
        
        # Occasionally miss a detection
        if detections and random.random() < 0.1:  # 10% chance to miss
            detections.pop()
        
        return detections

class OCRSimulator:
    """Simulates OCR barcode extraction"""
    
    def __init__(self):
        self.barcode_database = [
            "5901234123457", "9780201379624", "1234567890128",
            "4006381333931", "3661112507010", "5449000000996", 
            "3017620422003", "7613032620033", "8000500310427",
            "123456789012", "987654321098", "456123789045"
        ]
        self.success_rate = 0.8  # 80% success rate
    
    def extract_barcode(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Simulate barcode extraction from bag region"""
        # Simulate processing time
        time.sleep(0.01)
        
        if random.random() <= self.success_rate:
            return random.choice(self.barcode_database)
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
        
        # Pipeline components (all simulated)
        self.stapipy = StapipySimulator()
        self.yolo = YOLOSimulator(conf_threshold=config.get('confidence_threshold', 0.5))
        self.ocr = OCRSimulator()
        
        # State tracking
        self.current_frame = None
        self.current_frame_data = None
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
            self.current_frame_data = frame_data
            self.current_frame = frame_data['frame']
            
            # 2. YOLO Detection
            detections = self.yolo.detect(frame_data)
            
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
                timeout = self.config.get('last_bag_timeout', 3.0)
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
            stable_frames = self.config.get('stable_frames', 5)
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
                confidence=0.9,  # Simulated confidence
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
        self.current_frame_data = None
        self.frame_count = 0
        self.processing_times = []
        self.state_history = []
        self.last_detection_time = None
        self.stapipy = StapipySimulator()  # Reset simulator
        logger.info("Agent reset")

# Visualization functions using PIL only
def draw_pipeline_frame(frame_data: Dict, agent: BagDetectionAgent) -> Image.Image:
    """Draw pipeline information on frame"""
    # Convert numpy array back to PIL Image
    img = Image.fromarray(frame_data['frame'])
    draw = ImageDraw.Draw(img)
    
    status = agent.get_status()
    
    # Draw detection boxes
    if agent.context.all_detections:
        for detection in agent.context.all_detections[-5:]:  # Last 5 detections
            x1, y1, x2, y2 = detection.bbox
            
            # Color code: green for first, red for last, blue for others
            if detection.bag_id == status['first_bag']:
                color = (0, 255, 0)  # Green
            elif detection.bag_id == status['last_bag']:
                color = (255, 0, 0)  # Red
            else:
                color = (255, 255, 0)  # Yellow
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{detection.bag_id}: {detection.confidence:.2f}"
            try:
                font = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), label, font=font)
            except:
                bbox = (0, 0, len(label) * 6, 12)
            
            label_width = bbox[2] - bbox[0]
            draw.rectangle([x1, y1 - (bbox[3]-bbox[1]) - 5, x1 + label_width + 10, y1], 
                          fill=color)
            draw.text((x1 + 5, y1 - (bbox[3]-bbox[1]) - 2), label, fill=(0, 0, 0))
    
    # Draw status overlay
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([10, 10, 400, 250], fill=(0, 0, 0, 180))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Status text
    state_colors = {
        'WAITING_FIRST_BAG': (0, 255, 255),
        'FIRST_BAG_DETECTED': (0, 255, 0),
        'TRACKING_BAGS': (255, 255, 0),
        'WAITING_LAST_BAG': (255, 165, 0),
        'COMPLETED': (0, 255, 0),
        'ERROR': (255, 0, 0)
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
    ]
    
    y_pos = 30
    for line in lines:
        draw.text((20, y_pos), line, fill=color)
        y_pos += 25
    
    return img

# Streamlit App
def main():
    st.title("🛍️ Bag Detection Pipeline Agent")
    st.markdown("**Stapipy + YOLO + OCR Pipeline** - Manages 1st bag detection, barcode extraction, and last bag detection")
    
    # Initialize agent
    if 'agent' not in st.session_state:
        config = {
            'confidence_threshold': 0.5,
            'last_bag_timeout': 3.0,
            'stable_frames': 5
        }
        st.session_state.agent = BagDetectionAgent(config)
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Sidebar Configuration
    st.sidebar.header("Pipeline Configuration")
    
    st.sidebar.subheader("Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    last_bag_timeout = st.sidebar.slider("Last Bag Timeout (sec)", 1, 10, 3)
    
    # Update agent config
    st.session_state.agent.config.update({
        'confidence_threshold': confidence_threshold,
        'last_bag_timeout': last_bag_timeout
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
            
            if st.session_state.agent.current_frame_data is not None:
                # Draw visualization
                display_img = draw_pipeline_frame(
                    st.session_state.agent.current_frame_data, 
                    st.session_state.agent
                )
                
                video_placeholder.image(display_img, caption="Live Pipeline View", use_column_width=True)
            
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
            if st.session_state.agent.current_frame_data is not None:
                display_img = draw_pipeline_frame(
                    st.session_state.agent.current_frame_data, 
                    st.session_state.agent
                )
                video_placeholder.image(display_img, caption="Pipeline Ready", use_column_width=True)
            else:
                # Generate initial frame
                frame_data = st.session_state.agent.stapipy.get_frame()
                if frame_data:
                    st.session_state.agent.current_frame_data = frame_data
                    display_img = draw_pipeline_frame(frame_data, st.session_state.agent)
                    video_placeholder.image(display_img, caption="Pipeline Ready", use_column_width=True)
            
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
            st.metric("No Bag Frames", status['frames_without_bag'])
        
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
            avg_time = status['avg_processing_time'] * 1000
            st.metric("Avg Time", f"{avg_time:.1f}ms")
    
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
