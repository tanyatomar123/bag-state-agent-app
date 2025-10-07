"""
Bag Detection + OCR Streamlit App
---------------------------------
A lightweight state-machine agent that manages:
1️⃣ First bag detection
2️⃣ OCR barcode extraction
3️⃣ Last bag detection

Supports Stapipy (industrial camera) or mock camera for testing.
"""

import streamlit as st
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
from ultralytics import YOLO

# ==============================================================
# 🧩 Optional Stapipy import (auto fallback to mock mode)
# ==============================================================
try:
    from stapipy import StApi, StSystem
    STAPIPY_AVAILABLE = True
except ImportError:
    STAPIPY_AVAILABLE = False
    print("[INFO] Stapipy not found. Using mock frame generator instead.")


# ==============================================================
# 🧠 OCR placeholder (replace with Anant's OCR model)
# ==============================================================
def run_ocr_on_roi(roi):
    """
    Dummy OCR function — replace this with actual OCR model inference.
    """
    return {"text": "BAR" + str(np.random.randint(1000, 9999)), "score": 0.95}


# ==============================================================
# 🔄 Bag State Machine
# ==============================================================
class BagStateAgent:
    def __init__(self, yolo_model):
        self.model = yolo_model
        self.state = "IDLE"
        self.first_bag = None
        self.last_bag = None
        self.ocr_result = None
        self.running = False
        self.log = []
        self.session_results = []

    def log_event(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.log.append(entry)
        print(entry)

    def start(self):
        self.running = True
        self.state = "WAITING_FIRST"
        self.log_event("Agent started → WAITING_FIRST")

    def stop(self):
        self.running = False
        self.state = "IDLE"
        self.log_event("Agent stopped")

    def process_frame(self, frame):
        if not self.running:
            return frame

        results = self.model(frame, verbose=False)
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if self.state == "WAITING_FIRST":
                    self.first_bag = (x1, y1, x2, y2)
                    self.state = "OCR_PROCESSING"
                    self.log_event(f"🟢 First bag detected at {self.first_bag}")

                elif self.state == "OCR_PROCESSING":
                    roi = frame[y1:y2, x1:x2]
                    self.ocr_result = run_ocr_on_roi(roi)
                    self.state = "WAITING_LAST"
                    self.log_event(f"🔤 OCR result: {self.ocr_result}")

                elif self.state == "WAITING_LAST":
                    self.last_bag = (x1, y1, x2, y2)
                    self.log_event(f"🟣 Last bag detected at {self.last_bag}")
                    self.state = "COMPLETED"
                    self._save_session()
                    self.state = "WAITING_FIRST"
        return frame

    def _save_session(self):
        result = {
            "timestamp": datetime.now().isoformat(),
            "first_bag": self.first_bag,
            "last_bag": self.last_bag,
            "ocr_result": self.ocr_result,
        }
        self.session_results.append(result)
        self.log_event(f"✅ Session completed → {result}")


# ==============================================================
# 📷 Frame Sources (Stapipy or Mock)
# ==============================================================
def stapipy_stream():
    """
    Real camera streaming using Stapipy (if installed).
    """
    st_system = StSystem.CreateInstance()
    device = st_system.CreateFirstStDevice()
    stream = device.CreateStDataStream(0)
    stream.StartAcquisition()
    st_device = device

    while True:
        st_image = stream.WaitForFinishedBuffer(2000)
        if st_image is not None:
            frame = np.array(st_image.GetImageBuffer()).reshape(st_image.GetImageHeight(), st_image.GetImageWidth(), 3)
            stream.QueueBuffer(st_image)
            yield frame
        else:
            break


def mock_stream():
    """
    Generates random mock frames for testing (no camera needed).
    """
    while True:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        yield frame


# ==============================================================
# 🖥️ Streamlit UI
# ==============================================================
st.set_page_config(page_title="Bag State Machine Agent", layout="wide")
st.title("🎥 Bag Detection & OCR Flow Agent")

st.sidebar.header("⚙️ Controls")
start_btn = st.sidebar.button("▶️ Start Agent")
stop_btn = st.sidebar.button("⏹ Stop Agent")

if "agent" not in st.session_state:
    st.session_state.agent = BagStateAgent(YOLO("yolov8n.pt"))

agent = st.session_state.agent

frame_placeholder = st.empty()
log_placeholder = st.container()
session_placeholder = st.container()

frame_queue = queue.Queue(maxsize=1)


def camera_loop():
    """
    Threaded camera capture loop.
    """
    frame_source = stapipy_stream() if STAPIPY_AVAILABLE else mock_stream()

    for frame in frame_source:
        if not agent.running:
            break
        frame = cv2.resize(frame, (640, 480))
        processed = agent.process_frame(frame)
        if not frame_queue.full():
            frame_queue.put(processed)


if start_btn:
    if not agent.running:
        agent.start()
        threading.Thread(target=camera_loop, daemon=True).start()

if stop_btn:
    agent.stop()

# Display frames in real-time
if agent.running:
    if not frame_queue.empty():
        frame = frame_queue.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

# Display logs
with log_placeholder:
    st.markdown("### 🪵 Agent Log (Last 15 entries)")
    for line in agent.log[-15:]:
        st.text(line)

# Display session results
with session_placeholder:
    st.markdown("### 📦 Session Summary")
    if len(agent.session_results) > 0:
        st.dataframe(agent.session_results)
    else:
        st.info("No completed sessions yet.")
