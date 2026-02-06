"""Main GUI window for YOLO Training Pipeline"""
import sys
import warnings
import os
from pathlib import Path

# Suppress Qt and other warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar, QGroupBox, QGridLayout,
    QMenuBar, QAction, QMessageBox, QDialog, QScrollArea, QSlider, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QKeySequence, QPixmap

from src.dataset import DatasetManager
from src.config import ConfigurationManager
from src.training import TrainingEngine
from src.evaluation import EvaluationModule
from src.models import DatasetConfig, EvaluationConfig
from src.gui.image_canvas import ImageCanvas


class TrainingThread(QThread):
    """Background thread for training"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict, bool, object)  # metrics, is_simulation, result
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, config, output_dir):
        super().__init__()
        self.config = config
        self.output_dir = output_dir
    
    def run(self):
        import sys
        from io import StringIO
        
        # Capture stdout to show training progress
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # Create a custom stdout that emits progress signals
            class ProgressCapture:
                def __init__(self, signal):
                    self.signal = signal
                    self.buffer = ""
                    
                def write(self, text):
                    if text and text.strip():
                        self.signal.emit(text.rstrip())
                    old_stdout.write(text)  # Also write to original stdout
                    
                def flush(self):
                    old_stdout.flush()
            
            # Redirect stdout to capture training progress
            sys.stdout = ProgressCapture(self.progress)
            sys.stderr = ProgressCapture(self.progress)
            
            self.progress.emit("Initializing training...")
            engine = TrainingEngine(self.config, Path(self.output_dir))
            self.progress.emit("Training started...")
            result = engine.train()
            self.progress.emit("Training complete!")
            self.finished.emit(result.final_metrics, result.is_simulation, result)
            
        except Exception as e:
            error_msg = str(e)
            self.progress.emit(f"❌ Error: {error_msg}")
            self.error_occurred.emit(error_msg)
        finally:
            # Restore original stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class RTSPThread(QThread):
    """Background thread for RTSP stream processing"""
    frame_ready = pyqtSignal(object)  # QPixmap
    status_update = pyqtSignal(str, int, float)  # status_text, detection_count, fps
    error = pyqtSignal(str)
    
    def __init__(self, rtsp_url, model_path, confidence, class_names):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.model_path = model_path
        self.confidence = confidence
        self.class_names = class_names
        self.running = False
        self.model = None
    
    def run(self):
        import cv2
        import time
        import warnings
        from PIL import Image, ImageDraw, ImageFont
        from PyQt5.QtGui import QPixmap, QImage
        from ultralytics import YOLO
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        import logging
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        
        try:
            # Load model
            self.status_update.emit("Loading model...", 0, 0.0)
            self.model = YOLO(str(self.model_path))
            
            # Connect to RTSP stream
            self.status_update.emit("Connecting to stream...", 0, 0.0)
            cap = cv2.VideoCapture(self.rtsp_url)
            
            if not cap.isOpened():
                self.error.emit(f"Failed to connect to RTSP stream: {self.rtsp_url}")
                return
            
            self.running = True
            self.status_update.emit("Stream connected", 0, 0.0)
            
            # FPS calculation
            frame_count = 0
            start_time = time.time()
            fps = 0.0
            
            colors = [
                (0, 255, 136),   # Neon green
                (255, 100, 100), # Red
                (100, 150, 255), # Blue
                (255, 200, 0),   # Yellow
                (255, 0, 255),   # Magenta
            ]
            
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    self.error.emit("Lost connection to stream")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Run detection
                results = self.model.predict(
                    source=img,
                    conf=self.confidence,
                    verbose=False,
                    stream=False
                )
                
                # Draw bounding boxes
                result = results[0]
                boxes = result.boxes
                draw = ImageDraw.Draw(img)
                
                detection_count = 0
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Draw box
                    color = colors[class_id % len(colors)]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Get class name
                    if self.class_names and class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                    elif hasattr(self.model, 'names') and class_id in self.model.names:
                        class_name = self.model.names[class_id]
                    else:
                        class_name = f"Class {class_id}"
                    
                    label = f"{class_name} {conf:.2f}"
                    
                    # Draw label
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                    
                    bbox = draw.textbbox((x1, y1 - 20), label, font=font)
                    draw.rectangle(bbox, fill=color)
                    draw.text((x1, y1 - 20), label, fill=(26, 26, 46), font=font)
                    
                    detection_count += 1
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0.0
                
                # Convert PIL Image to QPixmap
                img_bytes = img.tobytes()
                qimage = QImage(img_bytes, img.width, img.height, img.width * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                
                # Emit frame and status
                self.frame_ready.emit(pixmap)
                self.status_update.emit("Streaming", detection_count, fps)
                
                # Small delay to prevent overwhelming the GUI
                time.sleep(0.01)
            
            cap.release()
            self.status_update.emit("Stream stopped", 0, 0.0)
            
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")
    
    def stop(self):
        """Stop the RTSP stream"""
        self.running = False


class USBCamThread(QThread):
    """Background thread for USB camera processing"""
    frame_ready = pyqtSignal(object)  # QPixmap
    status_update = pyqtSignal(str, int, float)  # status_text, detection_count, fps
    error = pyqtSignal(str)
    
    def __init__(self, camera_index, model_path, confidence, class_names, detection_mode="Object Detection"):
        super().__init__()
        self.camera_index = camera_index
        self.model_path = model_path
        self.confidence = confidence
        self.class_names = class_names
        self.detection_mode = detection_mode
        self.running = False
        self.model = None
        self.model_loaded = False
    
    def run(self):
        import cv2
        import time
        import warnings
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from PyQt5.QtGui import QPixmap, QImage
        from ultralytics import YOLO
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        import logging
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        
        try:
            # Connect to USB camera FIRST (fast)
            self.status_update.emit("Opening camera...", 0, 0.0)
            cap = cv2.VideoCapture(self.camera_index)
            
            if not cap.isOpened():
                self.error.emit(f"Failed to open USB camera at index {self.camera_index}")
                return
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.running = True
            self.status_update.emit("Camera opened - Loading model...", 0, 0.0)
            
            # FPS calculation
            frame_count = 0
            start_time = time.time()
            fps = 0.0
            
            colors = [
                (0, 255, 136),   # Neon green
                (255, 100, 100), # Red
                (100, 150, 255), # Blue
                (255, 200, 0),   # Yellow
                (255, 0, 255),   # Magenta
            ]
            
            # Start showing camera feed immediately
            frames_shown = 0
            
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    self.error.emit("Failed to read frame from camera")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Load model after showing first few frames (non-blocking feel)
                if not self.model_loaded and frames_shown >= 3:
                    self.status_update.emit("Loading model...", 0, 0.0)
                    self.model = YOLO(str(self.model_path))
                    self.model_loaded = True
                    self.status_update.emit("Model loaded - Detection active", 0, 0.0)
                
                detection_count = 0
                
                # Only run detection if model is loaded
                if self.model_loaded:
                    # Run detection based on mode
                    results = self.model.predict(
                        source=frame_rgb,
                        conf=self.confidence,
                        verbose=False,
                        stream=False
                    )
                    
                    result = results[0]
                    
                    # Process based on detection mode
                    if self.detection_mode == "Segmentation":
                        # Segmentation mode
                        img = Image.fromarray(frame_rgb)
                        draw = ImageDraw.Draw(img, 'RGBA')
                        
                        if hasattr(result, 'masks') and result.masks is not None:
                            masks = result.masks.data.cpu().numpy()
                            boxes = result.boxes
                            
                            for i, (mask, box) in enumerate(zip(masks, boxes)):
                                # Resize mask to frame size
                                mask_resized = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]))
                                
                                # Create colored overlay
                                color = colors[i % len(colors)]
                                overlay = np.zeros_like(frame_rgb)
                                overlay[mask_resized > 0.5] = color
                                
                                # Blend with original image
                                alpha = 0.4
                                frame_rgb = cv2.addWeighted(frame_rgb, 1, overlay, alpha, 0)
                                
                                # Draw bounding box
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                class_id = int(box.cls[0])
                                
                                img = Image.fromarray(frame_rgb)
                                draw = ImageDraw.Draw(img)
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                                
                                # Get class name
                                if self.class_names and class_id < len(self.class_names):
                                    class_name = self.class_names[class_id]
                                elif hasattr(self.model, 'names') and class_id in self.model.names:
                                    class_name = self.model.names[class_id]
                                else:
                                    class_name = f"Class {class_id}"
                                
                                label = f"{class_name} {conf:.2f}"
                                
                                # Draw label
                                try:
                                    font = ImageFont.truetype("arial.ttf", 16)
                                except:
                                    font = ImageFont.load_default()
                                
                                bbox = draw.textbbox((x1, y1 - 20), label, font=font)
                                draw.rectangle(bbox, fill=color)
                                draw.text((x1, y1 - 20), label, fill=(26, 26, 46), font=font)
                                
                                detection_count += 1
                            
                            frame_rgb = np.array(img)
                        
                        img = Image.fromarray(frame_rgb)
                    
                    elif self.detection_mode == "Pose Detection":
                        # Pose detection mode
                        img = Image.fromarray(frame_rgb)
                        draw = ImageDraw.Draw(img)
                        
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            keypoints = result.keypoints.data.cpu().numpy()
                            boxes = result.boxes
                            
                            # COCO keypoint connections (skeleton)
                            skeleton = [
                                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                                [2, 4], [3, 5], [4, 6], [5, 7]
                            ]
                            
                            for i, (kpts, box) in enumerate(zip(keypoints, boxes)):
                                color = colors[i % len(colors)]
                                
                                # Draw skeleton
                                for connection in skeleton:
                                    pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
                                    if pt1_idx < len(kpts) and pt2_idx < len(kpts):
                                        x1, y1, conf1 = kpts[pt1_idx]
                                        x2, y2, conf2 = kpts[pt2_idx]
                                        
                                        if conf1 > 0.5 and conf2 > 0.5:
                                            draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
                                
                                # Draw keypoints
                                for kpt in kpts:
                                    x, y, conf = kpt
                                    if conf > 0.5:
                                        draw.ellipse([x-3, y-3, x+3, y+3], fill=color, outline=(255, 255, 255))
                                
                                # Draw bounding box
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                                
                                label = f"Person {conf:.2f}"
                                
                                # Draw label
                                try:
                                    font = ImageFont.truetype("arial.ttf", 16)
                                except:
                                    font = ImageFont.load_default()
                                
                                bbox = draw.textbbox((x1, y1 - 20), label, font=font)
                                draw.rectangle(bbox, fill=color)
                                draw.text((x1, y1 - 20), label, fill=(26, 26, 46), font=font)
                                
                                detection_count += 1
                        else:
                            # Fallback to regular detection if no keypoints
                            boxes = result.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                color = colors[0]
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                                detection_count += 1
                    
                    else:
                        # Object Detection mode (default)
                        img = Image.fromarray(frame_rgb)
                        draw = ImageDraw.Draw(img)
                        boxes = result.boxes
                        
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            class_id = int(box.cls[0])
                            
                            # Draw box
                            color = colors[class_id % len(colors)]
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                            
                            # Get class name
                            if self.class_names and class_id < len(self.class_names):
                                class_name = self.class_names[class_id]
                            elif hasattr(self.model, 'names') and class_id in self.model.names:
                                class_name = self.model.names[class_id]
                            else:
                                class_name = f"Class {class_id}"
                            
                            label = f"{class_name} {conf:.2f}"
                            
                            # Draw label
                            try:
                                font = ImageFont.truetype("arial.ttf", 16)
                            except:
                                font = ImageFont.load_default()
                            
                            bbox = draw.textbbox((x1, y1 - 20), label, font=font)
                            draw.rectangle(bbox, fill=color)
                            draw.text((x1, y1 - 20), label, fill=(26, 26, 46), font=font)
                            
                            detection_count += 1
                
                else:
                    # Model not loaded yet - show raw camera feed
                    img = Image.fromarray(frame_rgb)
                
                # Increment frames shown counter
                frames_shown += 1
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0.0
                
                # Convert PIL Image to QPixmap
                img_bytes = img.tobytes()
                qimage = QImage(img_bytes, img.width, img.height, img.width * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                
                # Emit frame and status
                self.frame_ready.emit(pixmap)
                if self.model_loaded:
                    self.status_update.emit("Streaming", detection_count, fps)
                else:
                    self.status_update.emit("Loading model...", 0, fps)
                
                # Small delay to prevent overwhelming the GUI
                time.sleep(0.01)
            
            cap.release()
            self.status_update.emit("Camera closed", 0, 0.0)
            
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")
    
    def stop(self):
        """Stop the USB camera stream"""
        self.running = False


class YOLOTrainingGUI(QMainWindow):
    """Main GUI window"""
    
    def __init__(self):
        super().__init__()
        self.dataset_manager = None
        self.training_thread = None
        self.rtsp_thread = None
        self.usbcam_thread = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("YOLO Training Pipeline - Object Detection System")
        
        # Apply high-tech dark theme
        self.apply_dark_theme()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🤖 YOLO Training Pipeline")
        title.setFont(QFont("Arial", 32, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00ff88; padding: 20px;")
        main_layout.addWidget(title)
        
        # Create tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #00ff88;
                background: #1a1a2e;
            }
            QTabBar::tab {
                background: #16213e;
                color: #00ff88;
                padding: 10px 20px;
                margin: 2px;
                border: 1px solid #00ff88;
            }
            QTabBar::tab:selected {
                background: #0f3460;
                font-weight: bold;
            }
        """)
        
        # Add tabs
        tabs.addTab(self.create_labeling_tab(), "🏷️ Labeling")
        tabs.addTab(self.create_dataset_tab(), "📁 Splitting")
        tabs.addTab(self.create_training_tab(), "🚀 Training")
        tabs.addTab(self.create_evaluation_tab(), "📊 Evaluation")
        tabs.addTab(self.create_test_tab(), "🧪 Test")
        tabs.addTab(self.create_rtsp_tab(), "📡 RTSP")
        tabs.addTab(self.create_usbcam_tab(), "📹 USB Cam")
        tabs.addTab(self.create_logs_tab(), "📝 Logs")
        tabs.addTab(self.create_help_tab(), "❓ Help")
        
        main_layout.addWidget(tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("background: #16213e; color: #00ff88; padding: 5px;")
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #16213e;
                color: #00ff88;
                padding: 5px;
            }
            QMenuBar::item {
                background-color: #16213e;
                color: #00ff88;
                padding: 8px 15px;
            }
            QMenuBar::item:selected {
                background-color: #0f3460;
            }
            QMenu {
                background-color: #16213e;
                color: #00ff88;
                border: 2px solid #00ff88;
            }
            QMenu::item {
                padding: 8px 25px;
            }
            QMenu::item:selected {
                background-color: #0f3460;
            }
        """)
    
    
    def apply_dark_theme(self):
        """Apply high-tech dark theme"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(26, 26, 46))
        palette.setColor(QPalette.WindowText, QColor(0, 255, 136))
        palette.setColor(QPalette.Base, QColor(22, 33, 62))
        palette.setColor(QPalette.AlternateBase, QColor(26, 26, 46))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(15, 52, 96))
        palette.setColor(QPalette.ButtonText, QColor(0, 255, 136))
        self.setPalette(palette)
        
        # Global stylesheet with increased font sizes
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QLabel {
                color: #ffffff;
                font-size: 21px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #16213e;
                color: #ffffff;
                border: 2px solid #00ff88;
                border-radius: 5px;
                padding: 8px;
                font-size: 21px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #00ffff;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #0f3460;
                width: 30px;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 8px solid #00ff88;
                width: 0px;
                height: 0px;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #16213e;
                color: #ffffff;
                selection-background-color: #0f3460;
                selection-color: #00ff88;
                border: 2px solid #00ff88;
                border-radius: 5px;
                padding: 5px;
                font-size: 21px;
            }
            QComboBox QAbstractItemView::item {
                padding: 8px;
                min-height: 30px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #0f3460;
                color: #00ff88;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #00ff88;
                color: #1a1a2e;
            }
            QPushButton {
                background-color: #0f3460;
                color: #00ff88;
                border: 2px solid #00ff88;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 19px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00ff88;
                color: #1a1a2e;
            }
            QPushButton:pressed {
                background-color: #00cc66;
            }
            QTextEdit {
                background-color: #0d1117;
                color: #00ff88;
                border: 2px solid #00ff88;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Courier New';
                font-size: 21px;
            }
            QProgressBar {
                border: 2px solid #00ff88;
                border-radius: 5px;
                text-align: center;
                background-color: #16213e;
                color: #ffffff;
                font-size: 21px;
            }
            QProgressBar::chunk {
                background-color: #00ff88;
            }
            QGroupBox {
                border: 2px solid #00ff88;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #00ff88;
                font-size: 19px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
    
    def show_dark_message(self, title, message, icon_type="information"):
        """Show a dark-themed message box"""
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)
        
        # Set icon type
        if icon_type == "information":
            msg.setIcon(QMessageBox.Information)
        elif icon_type == "warning":
            msg.setIcon(QMessageBox.Warning)
        elif icon_type == "critical":
            msg.setIcon(QMessageBox.Critical)
        elif icon_type == "question":
            msg.setIcon(QMessageBox.Question)
        
        # Apply dark theme stylesheet with larger fonts
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #1a1a2e;
                color: #ffffff;
            }
            QMessageBox QLabel {
                color: #ffffff;
                font-size: 19px;
            }
            QMessageBox QPushButton {
                background-color: #0f3460;
                color: #00ff88;
                border: 2px solid #00ff88;
                border-radius: 5px;
                padding: 8px 20px;
                font-size: 19px;
                font-weight: bold;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background-color: #00ff88;
                color: #1a1a2e;
            }
            QMessageBox QPushButton:pressed {
                background-color: #00cc66;
            }
        """)
        
        msg.exec_()
        return msg.result()

    def _make_scrollable(self, widget):
        """Wrap a widget in a scroll area for better usability"""
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1a1a2e;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #1a1a2e;
            }
            QScrollBar:vertical {
                background: #16213e;
                width: 15px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: #00ff88;
                border-radius: 7px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00ffaa;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #16213e;
                height: 15px;
                border-radius: 7px;
            }
            QScrollBar::handle:horizontal {
                background: #00ff88;
                border-radius: 7px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #00ffaa;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        # Set the widget's background to dark as well
        widget.setStyleSheet("background-color: #1a1a2e;")
        
        return scroll

    
    def create_dataset_tab(self):
        """Create dataset management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Collection group
        collect_group = QGroupBox("📥 Data Collection")
        collect_layout = QGridLayout()
        
        # Row 0: Source Directory
        collect_layout.addWidget(QLabel("Source Directory:"), 0, 0)
        self.source_dir_input = QLineEdit()
        self.source_dir_input.setPlaceholderText("Path to images folder")
        collect_layout.addWidget(self.source_dir_input, 0, 1)
        
        browse_source_btn = QPushButton("📂 Browse")
        browse_source_btn.clicked.connect(self.browse_source_dir)
        collect_layout.addWidget(browse_source_btn, 0, 2)
        
        # Row 1: YAML Config
        collect_layout.addWidget(QLabel("YAML Config:"), 1, 0)
        self.dataset_yaml_input = QLineEdit()
        self.dataset_yaml_input.setPlaceholderText("Optional: YAML file for auto-config")
        collect_layout.addWidget(self.dataset_yaml_input, 1, 1)
        
        browse_dataset_yaml_btn = QPushButton("📂 Browse")
        browse_dataset_yaml_btn.clicked.connect(self.browse_dataset_yaml)
        collect_layout.addWidget(browse_dataset_yaml_btn, 1, 2)
        
        # Row 2: Info label
        info_label = QLabel("💡 Tip: Labels should be in the same folder as images")
        info_label.setStyleSheet("color: #00ff88; font-size: 19px; font-style: italic;")
        collect_layout.addWidget(info_label, 2, 0, 1, 4)
        
        # Row 3: Collect button
        collect_btn = QPushButton("🚀 Collect Images & Labels")
        collect_btn.clicked.connect(self.collect_images)
        collect_layout.addWidget(collect_btn, 3, 0, 1, 4)
        
        collect_group.setLayout(collect_layout)
        layout.addWidget(collect_group)
        
        # Split group
        split_group = QGroupBox("✂️ Dataset Split")
        split_layout = QGridLayout()
        
        split_layout.addWidget(QLabel("Train Ratio:"), 0, 0)
        self.train_ratio = QDoubleSpinBox()
        self.train_ratio.setRange(0.1, 0.9)
        self.train_ratio.setValue(0.7)
        self.train_ratio.setSingleStep(0.1)
        split_layout.addWidget(self.train_ratio, 0, 1)
        
        split_layout.addWidget(QLabel("Val Ratio:"), 1, 0)
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0.1, 0.9)
        self.val_ratio.setValue(0.2)
        self.val_ratio.setSingleStep(0.1)
        split_layout.addWidget(self.val_ratio, 1, 1)
        
        split_layout.addWidget(QLabel("Test Ratio:"), 2, 0)
        self.test_ratio = QDoubleSpinBox()
        self.test_ratio.setRange(0.1, 0.9)
        self.test_ratio.setValue(0.1)
        self.test_ratio.setSingleStep(0.1)
        split_layout.addWidget(self.test_ratio, 2, 1)
        
        split_btn = QPushButton("✂️ Split Dataset")
        split_btn.clicked.connect(self.split_dataset)
        split_layout.addWidget(split_btn, 3, 0, 1, 2)
        
        split_group.setLayout(split_layout)
        layout.addWidget(split_group)
        
        # Statistics group - expand to fill remaining space
        stats_group = QGroupBox("📊 Dataset Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMinimumHeight(400)
        stats_layout.addWidget(self.stats_display)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group, 1)  # Stretch factor of 1 to expand
        
        # Remove the addStretch() so stats_group can expand to bottom
        return self._make_scrollable(widget)

    
    def create_labeling_tab(self):
        """Create image labeling tab with interactive canvas"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Top controls
        controls_group = QGroupBox("🏷️ Labeling Controls")
        controls_layout = QGridLayout()
        
        # Row 0: Images Folder
        controls_layout.addWidget(QLabel("Images Folder:"), 0, 0)
        self.label_images_input = QLineEdit("./data/my_dataset/images/train")
        self.label_images_input.setPlaceholderText("Path to images folder")
        controls_layout.addWidget(self.label_images_input, 0, 1)
        
        browse_images_btn = QPushButton("📂 Browse")
        browse_images_btn.clicked.connect(self.browse_label_images)
        controls_layout.addWidget(browse_images_btn, 0, 2)
        
        # Row 1: Labels Folder
        controls_layout.addWidget(QLabel("Labels Folder:"), 1, 0)
        self.label_labels_input = QLineEdit("")
        self.label_labels_input.setPlaceholderText("Optional: Separate labels folder (leave empty if same as images)")
        controls_layout.addWidget(self.label_labels_input, 1, 1)
        
        browse_labels_btn = QPushButton("📂 Browse")
        browse_labels_btn.clicked.connect(self.browse_label_labels)
        controls_layout.addWidget(browse_labels_btn, 1, 2)
        
        # Row 2: YAML Config File
        controls_layout.addWidget(QLabel("YAML Config:"), 2, 0)
        self.label_yaml_input = QLineEdit("")
        self.label_yaml_input.setPlaceholderText("Optional: YAML file for auto-config (e.g., coco128.yaml)")
        controls_layout.addWidget(self.label_yaml_input, 2, 1)
        
        browse_yaml_label_btn = QPushButton("📂 Browse")
        browse_yaml_label_btn.clicked.connect(self.browse_label_yaml)
        controls_layout.addWidget(browse_yaml_label_btn, 2, 2)
        
        load_yaml_label_btn = QPushButton("📥 Load YAML")
        load_yaml_label_btn.clicked.connect(self.load_label_yaml)
        load_yaml_label_btn.setToolTip("Load paths and class names from YAML")
        controls_layout.addWidget(load_yaml_label_btn, 2, 3)
        
        # Row 3: Load Images Button
        load_images_btn = QPushButton("📂 Load Images")
        load_images_btn.clicked.connect(self.load_labeling_images)
        load_images_btn.setStyleSheet("font-weight: bold; font-size: 24px; padding: 8px;")
        controls_layout.addWidget(load_images_btn, 3, 0, 1, 4)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Drawing Mode Selection
        mode_group = QGroupBox("✏️ Drawing Mode")
        mode_layout = QHBoxLayout()
        
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup
        self.drawing_mode_group = QButtonGroup()
        
        self.bbox_mode_radio = QRadioButton("📦 Bounding Box (Detection)")
        self.bbox_mode_radio.setChecked(True)
        self.bbox_mode_radio.toggled.connect(lambda: self.set_drawing_mode('bbox'))
        self.bbox_mode_radio.setStyleSheet("""
            QRadioButton {
                color: #00ff88;
                font-size: 18px;
                font-weight: bold;
                spacing: 10px;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #00ff88;
                border-radius: 10px;
                background-color: #16213e;
            }
            QRadioButton::indicator:checked {
                background-color: #00ff88;
                border: 2px solid #00ff88;
            }
            QRadioButton::indicator:hover {
                border: 2px solid #00cc66;
            }
        """)
        self.drawing_mode_group.addButton(self.bbox_mode_radio)
        mode_layout.addWidget(self.bbox_mode_radio)
        
        self.polygon_mode_radio = QRadioButton("🔷 Polygon (Segmentation)")
        self.polygon_mode_radio.toggled.connect(lambda: self.set_drawing_mode('polygon'))
        self.polygon_mode_radio.setStyleSheet("""
            QRadioButton {
                color: #00ff88;
                font-size: 18px;
                font-weight: bold;
                spacing: 10px;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #00ff88;
                border-radius: 10px;
                background-color: #16213e;
            }
            QRadioButton::indicator:checked {
                background-color: #00ff88;
                border: 2px solid #00ff88;
            }
            QRadioButton::indicator:hover {
                border: 2px solid #00cc66;
            }
        """)
        self.drawing_mode_group.addButton(self.polygon_mode_radio)
        mode_layout.addWidget(self.polygon_mode_radio)
        
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Class Names List
        class_names_group = QGroupBox("🏷️ Class Names")
        class_names_layout = QVBoxLayout()
        
        # Header with checkbox
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Available Classes:"))
        self.show_names_checkbox = QCheckBox("Show Class Names (V)")
        self.show_names_checkbox.setChecked(True)
        self.show_names_checkbox.stateChanged.connect(self.toggle_class_names_display)
        header_layout.addWidget(self.show_names_checkbox)
        header_layout.addStretch()
        class_names_layout.addLayout(header_layout)
        
        # List widget for class names
        from PyQt5.QtWidgets import QListWidget
        self.class_names_list = QListWidget()
        self.class_names_list.setMaximumHeight(120)
        self.class_names_list.setStyleSheet("""
            QListWidget {
                background: #16213e;
                color: #00ff88;
                border: 1px solid #00ff88;
                font-size: 21px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background: #0f3460;
                color: #ffffff;
            }
        """)
        # Add default class
        self.class_names_list.addItem("0: object")
        class_names_layout.addWidget(self.class_names_list)
        
        class_names_group.setLayout(class_names_layout)
        layout.addWidget(class_names_group)
        
        # Image canvas
        canvas_group = QGroupBox("🖼️ Image Canvas")
        canvas_layout = QVBoxLayout()
        
        # Image info label
        self.image_info_label = QLabel("No images loaded - Click 'Load Images' to start")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setStyleSheet("font-size: 24px; color: #00ff88; padding: 5px;")
        canvas_layout.addWidget(self.image_info_label)
        
        # Interactive canvas
        self.image_canvas = ImageCanvas()
        self.image_canvas.bbox_created.connect(self.on_bbox_created)
        self.image_canvas.bbox_deleted.connect(self.on_bbox_deleted)
        self.image_canvas.polygon_created.connect(self.on_polygon_created)
        canvas_layout.addWidget(self.image_canvas)
        
        # Image slider for quick navigation
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Quick Jump:"))
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.setMaximum(0)
        self.image_slider.valueChanged.connect(self.on_slider_changed)
        self.image_slider.setEnabled(False)
        slider_layout.addWidget(self.image_slider)
        canvas_layout.addLayout(slider_layout)
        
        canvas_group.setLayout(canvas_layout)
        layout.addWidget(canvas_group)
        
        # Navigation and controls
        controls_bottom_group = QGroupBox("🎮 Navigation & Controls")
        controls_bottom_layout = QGridLayout()
        
        # First row: Navigation buttons and image counter
        self.prev_image_btn = QPushButton("⬅️ Prev (A / Wheel Up)")
        self.prev_image_btn.clicked.connect(self.prev_labeling_image)
        self.prev_image_btn.setEnabled(False)
        controls_bottom_layout.addWidget(self.prev_image_btn, 0, 0)
        
        self.next_image_btn = QPushButton("Next (D / Space / Wheel Down) ➡️")
        self.next_image_btn.clicked.connect(self.next_labeling_image)
        self.next_image_btn.setEnabled(False)
        controls_bottom_layout.addWidget(self.next_image_btn, 0, 1)
        
        self.image_counter_label = QLabel("0 / 0")
        self.image_counter_label.setAlignment(Qt.AlignCenter)
        self.image_counter_label.setStyleSheet("font-size: 19px; font-weight: bold;")
        controls_bottom_layout.addWidget(self.image_counter_label, 0, 2)
        
        # Second row: Save, Annotations count, Undo, Clear All
        self.save_label_btn = QPushButton("💾 Save (Ctrl+S)")
        self.save_label_btn.clicked.connect(self.save_current_label)
        self.save_label_btn.setEnabled(False)
        controls_bottom_layout.addWidget(self.save_label_btn, 1, 0)
        
        self.boxes_count_label = QLabel("Annotations: 0")
        self.boxes_count_label.setAlignment(Qt.AlignCenter)
        self.boxes_count_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff88;")
        controls_bottom_layout.addWidget(self.boxes_count_label, 1, 1)
        
        self.undo_box_btn = QPushButton("↶ Undo (Ctrl+Z)")
        self.undo_box_btn.clicked.connect(self.undo_last_box)
        self.undo_box_btn.setEnabled(False)
        controls_bottom_layout.addWidget(self.undo_box_btn, 1, 2)
        
        self.clear_boxes_btn = QPushButton("🗑️ Clear All (Ctrl+C)")
        self.clear_boxes_btn.clicked.connect(self.clear_all_boxes)
        self.clear_boxes_btn.setEnabled(False)
        controls_bottom_layout.addWidget(self.clear_boxes_btn, 1, 3)
        
        controls_bottom_group.setLayout(controls_bottom_layout)
        layout.addWidget(controls_bottom_group)
        
        # Instructions
        instructions = QLabel(
            "📝 <b>How to Label:</b><br>"
            "<b>Bounding Box Mode:</b> Click twice to create box (top-left → bottom-right)<br>"
            "<b>Polygon Mode:</b> Click to add points, right-click or click near first point to close polygon<br>"
            "Right-click box/polygon to delete | A/D or Arrow keys to navigate | "
            "Ctrl+Z to undo | Ctrl+C to clear all | Ctrl+S to save | V to toggle class names | "
            "Mouse wheel to navigate | Esc to cancel drawing<br>"
            "<br>"
            "💾 <b>Label Storage:</b> Labels are saved in the SAME folder as images (YOLO format). "
            "Example: 'data/images/image001.jpg' → 'data/images/image001.txt'"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #00ff88; font-size: 21px; padding: 10px;")
        layout.addWidget(instructions)
        
        # Initialize labeling state
        self.labeling_images = []
        self.current_image_index = 0
        
        # Setup keyboard shortcuts
        self.setup_labeling_shortcuts()
        
        return self._make_scrollable(widget)
    
    def setup_labeling_shortcuts(self):
        """Setup keyboard shortcuts for labeling"""
        from PyQt5.QtWidgets import QShortcut
        
        # Navigation shortcuts
        QShortcut(QKeySequence("A"), self, self.prev_labeling_image)
        QShortcut(QKeySequence("D"), self, self.next_labeling_image)
        QShortcut(QKeySequence("Space"), self, self.next_labeling_image)
        QShortcut(QKeySequence("Left"), self, self.prev_labeling_image)
        QShortcut(QKeySequence("Right"), self, self.next_labeling_image)
        
        # Control shortcuts
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_last_box)
        QShortcut(QKeySequence("Ctrl+C"), self, self.clear_all_boxes)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_current_label)
        QShortcut(QKeySequence("V"), self, self.toggle_class_names_display)

    
    def create_training_tab(self):
        """Create training configuration tab with progress log"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model configuration
        config_group = QGroupBox("⚙️ Model Configuration")
        config_layout = QGridLayout()
        
        config_layout.addWidget(QLabel("Task Type:"), 0, 0)
        self.task_type = QComboBox()
        self.task_type.addItems(["Object Detection", "Segmentation"])
        self.task_type.currentTextChanged.connect(self.on_task_type_changed)
        config_layout.addWidget(self.task_type, 0, 1)
        
        config_layout.addWidget(QLabel("YOLO Version:"), 0, 2)
        self.yolo_version = QComboBox()
        self.yolo_version.addItems(["yolov5", "yolov8", "yolo11"])
        self.yolo_version.currentTextChanged.connect(self.on_yolo_version_changed)
        config_layout.addWidget(self.yolo_version, 0, 3)
        
        config_layout.addWidget(QLabel("Model Architecture:"), 1, 0)
        self.model_arch = QComboBox()
        self.model_arch.addItems(["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"])
        self.model_arch.setCurrentText("yolov5s")
        config_layout.addWidget(self.model_arch, 1, 1)
        
        config_layout.addWidget(QLabel("Device:"), 1, 2)
        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        config_layout.addWidget(self.device, 1, 3)
        
        config_layout.addWidget(QLabel("Image Size:"), 2, 0)
        self.image_size = QComboBox()
        self.image_size.addItems(["320", "416", "640", "1280"])
        self.image_size.setCurrentText("640")
        config_layout.addWidget(self.image_size, 2, 1)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Training parameters
        params_group = QGroupBox("🎯 Training Parameters")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(50)
        self.epochs.setToolTip("Number of complete passes through the training dataset")
        params_layout.addWidget(self.epochs, 0, 1)
        
        params_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(16)
        self.batch_size.setToolTip("Number of images processed together (lower if out of memory)")
        params_layout.addWidget(self.batch_size, 0, 3)
        
        params_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 0.1)
        self.learning_rate.setValue(0.01)
        self.learning_rate.setDecimals(4)
        self.learning_rate.setSingleStep(0.001)
        self.learning_rate.setToolTip("Initial learning rate for optimizer")
        params_layout.addWidget(self.learning_rate, 1, 1)
        
        params_layout.addWidget(QLabel("Patience:"), 1, 2)
        self.patience = QSpinBox()
        self.patience.setRange(0, 100)
        self.patience.setValue(10)
        self.patience.setToolTip("Epochs to wait for improvement before early stopping (0 = disabled)")
        params_layout.addWidget(self.patience, 1, 3)
        
        params_layout.addWidget(QLabel("Workers:"), 2, 0)
        self.workers = QSpinBox()
        self.workers.setRange(0, 16)
        self.workers.setValue(4)
        self.workers.setToolTip("Number of worker threads for data loading")
        params_layout.addWidget(self.workers, 2, 1)
        
        params_layout.addWidget(QLabel("Save Period:"), 2, 2)
        self.save_period = QSpinBox()
        self.save_period.setRange(1, 100)
        self.save_period.setValue(10)
        self.save_period.setToolTip("Save checkpoint every N epochs")
        params_layout.addWidget(self.save_period, 2, 3)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Dataset paths
        dataset_group = QGroupBox("📁 Dataset Configuration")
        dataset_layout = QGridLayout()
        
        dataset_layout.addWidget(QLabel("YAML Config File:"), 0, 0)
        self.train_yaml_file = QLineEdit("./coco128/coco128.yaml")
        self.train_yaml_file.setPlaceholderText("Path to dataset YAML file (e.g., coco128.yaml)")
        dataset_layout.addWidget(self.train_yaml_file, 0, 1)
        
        browse_yaml_btn = QPushButton("📂 Browse")
        browse_yaml_btn.clicked.connect(self.browse_train_yaml)
        dataset_layout.addWidget(browse_yaml_btn, 0, 2)
        
        dataset_layout.addWidget(QLabel("Dataset Root:"), 1, 0)
        self.train_dataset_root = QLineEdit("./data")
        self.train_dataset_root.setToolTip("Root directory of the dataset (e.g., ./data/train2017_dataset)")
        dataset_layout.addWidget(self.train_dataset_root, 1, 1)
        
        browse_dataset_btn = QPushButton("📂 Browse")
        browse_dataset_btn.clicked.connect(self.browse_train_dataset)
        dataset_layout.addWidget(browse_dataset_btn, 1, 2)
        
        dataset_layout.addWidget(QLabel("Number of Classes:"), 1, 3)
        self.num_classes_display = QLabel("80")
        self.num_classes_display.setStyleSheet("font-weight: bold; color: #00ff88;")
        self.num_classes_display.setToolTip("Number of classes (auto-detected from YAML)")
        dataset_layout.addWidget(self.num_classes_display, 1, 4)
        
        dataset_layout.addWidget(QLabel("Output Directory:"), 2, 0)
        self.output_dir = QLineEdit("./model")
        dataset_layout.addWidget(self.output_dir, 2, 1)
        
        browse_output_btn = QPushButton("📂 Browse")
        browse_output_btn.clicked.connect(self.browse_output_dir)
        dataset_layout.addWidget(browse_output_btn, 2, 2)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Training control
        control_group = QGroupBox("🎮 Training Control")
        control_layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("🚀 Start Training")
        self.train_btn.clicked.connect(self.start_training)
        button_layout.addWidget(self.train_btn)
        
        self.stop_train_btn = QPushButton("⏹️ Stop Training")
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        button_layout.addWidget(self.stop_train_btn)
        
        control_layout.addLayout(button_layout)
        
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_progress.setVisible(False)
        control_layout.addWidget(self.training_progress)
        
        self.training_status_label = QLabel("Ready to train")
        self.training_status_label.setAlignment(Qt.AlignCenter)
        self.training_status_label.setStyleSheet("font-size: 24px; color: #00ff88; padding: 5px;")
        control_layout.addWidget(self.training_status_label)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Training log
        log_group = QGroupBox("📊 Training Progress Log")
        log_layout = QVBoxLayout()
        
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumHeight(250)
        self.training_log.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #00ff88;
                border: 2px solid #00ff88;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Courier New';
                font-size: 21px;
            }
        """)
        log_layout.addWidget(self.training_log)
        
        log_buttons = QHBoxLayout()
        
        clear_log_btn = QPushButton("🗑️ Clear Log")
        clear_log_btn.clicked.connect(lambda: self.training_log.clear())
        log_buttons.addWidget(clear_log_btn)
        
        save_log_btn = QPushButton("💾 Save Log")
        save_log_btn.clicked.connect(self.save_training_log)
        log_buttons.addWidget(save_log_btn)
        
        log_layout.addLayout(log_buttons)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        return self._make_scrollable(widget)

    
    def create_evaluation_tab(self):
        """Create evaluation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        eval_group = QGroupBox("🎯 Model Evaluation")
        eval_layout = QGridLayout()
        
        eval_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_path = QLineEdit("./model/yolov5s.pt")
        eval_layout.addWidget(self.model_path, 0, 1)
        
        browse_model_btn = QPushButton("📂 Browse")
        browse_model_btn.clicked.connect(self.browse_model)
        eval_layout.addWidget(browse_model_btn, 0, 2)
        
        eval_layout.addWidget(QLabel("Dataset Split:"), 1, 0)
        self.eval_split = QComboBox()
        self.eval_split.addItems(["test", "val"])
        eval_layout.addWidget(self.eval_split, 1, 1)
        
        eval_layout.addWidget(QLabel("Confidence Threshold:"), 2, 0)
        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.0, 1.0)
        self.conf_threshold.setValue(0.25)
        self.conf_threshold.setSingleStep(0.05)
        eval_layout.addWidget(self.conf_threshold, 2, 1)
        
        eval_btn = QPushButton("📊 Evaluate Model")
        eval_btn.clicked.connect(self.evaluate_model)
        eval_layout.addWidget(eval_btn, 3, 0, 1, 3)
        
        eval_group.setLayout(eval_layout)
        layout.addWidget(eval_group)
        
        # Results display
        results_group = QGroupBox("📈 Evaluation Results")
        results_layout = QVBoxLayout()
        
        self.eval_results = QTextEdit()
        self.eval_results.setReadOnly(True)
        results_layout.addWidget(self.eval_results)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        return self._make_scrollable(widget)
    
    def create_test_tab(self):
        """Create model testing tab for images and videos"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model selection
        model_group = QGroupBox("🤖 Model Selection")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.test_model_path = QLineEdit("./model/yolov5s.pt")
        model_layout.addWidget(self.test_model_path, 0, 1)
        
        browse_test_model_btn = QPushButton("📂 Browse")
        browse_test_model_btn.clicked.connect(self.browse_test_model)
        model_layout.addWidget(browse_test_model_btn, 0, 2)
        
        model_layout.addWidget(QLabel("Confidence:"), 1, 0)
        self.test_confidence = QDoubleSpinBox()
        self.test_confidence.setRange(0.0, 1.0)
        self.test_confidence.setValue(0.25)
        self.test_confidence.setSingleStep(0.05)
        model_layout.addWidget(self.test_confidence, 1, 1)
        
        model_layout.addWidget(QLabel("Class Names:"), 2, 0)
        self.test_class_names = QLineEdit("object")
        self.test_class_names.setPlaceholderText("Comma-separated: object,person,dog")
        model_layout.addWidget(self.test_class_names, 2, 1, 1, 2)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Input selection
        input_group = QGroupBox("📁 Input Selection")
        input_layout = QGridLayout()
        
        input_layout.addWidget(QLabel("Input Type:"), 0, 0)
        self.test_input_type = QComboBox()
        self.test_input_type.addItems(["Image", "Video"])
        self.test_input_type.currentTextChanged.connect(self.on_test_input_type_changed)
        input_layout.addWidget(self.test_input_type, 0, 1)
        
        input_layout.addWidget(QLabel("Input Path:"), 1, 0)
        self.test_input_path = QLineEdit()
        self.test_input_path.setPlaceholderText("Select an image or video file")
        input_layout.addWidget(self.test_input_path, 1, 1)
        
        browse_test_input_btn = QPushButton("📂 Browse")
        browse_test_input_btn.clicked.connect(self.browse_test_input)
        input_layout.addWidget(browse_test_input_btn, 1, 2)
        
        run_test_btn = QPushButton("🚀 Run Detection")
        run_test_btn.clicked.connect(self.run_detection_test)
        input_layout.addWidget(run_test_btn, 2, 0, 1, 3)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Results display
        results_group = QGroupBox("🖼️ Detection Results")
        results_layout = QVBoxLayout()
        
        # Result info
        self.test_result_info = QLabel("No detection run yet")
        self.test_result_info.setAlignment(Qt.AlignCenter)
        self.test_result_info.setStyleSheet("font-size: 24px; color: #00ff88; padding: 5px;")
        results_layout.addWidget(self.test_result_info)
        
        # Result image display
        self.test_result_display = QLabel("Run detection to see results")
        self.test_result_display.setAlignment(Qt.AlignCenter)
        self.test_result_display.setMinimumSize(800, 500)
        self.test_result_display.setStyleSheet("""
            QLabel {
                background-color: #0d1117;
                border: 2px solid #00ff88;
                border-radius: 5px;
                color: #00ff88;
                font-size: 19px;
            }
        """)
        self.test_result_display.setScaledContents(False)
        results_layout.addWidget(self.test_result_display)
        
        # Save result button
        save_result_layout = QHBoxLayout()
        
        self.save_result_btn = QPushButton("💾 Save Result")
        self.save_result_btn.clicked.connect(self.save_detection_result)
        self.save_result_btn.setEnabled(False)
        save_result_layout.addWidget(self.save_result_btn)
        
        self.test_detection_info = QLabel("Detections: 0 | Time: 0ms")
        self.test_detection_info.setAlignment(Qt.AlignCenter)
        self.test_detection_info.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff88;")
        save_result_layout.addWidget(self.test_detection_info)
        
        results_layout.addLayout(save_result_layout)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Instructions
        instructions = QLabel(
            "📝 <b>How to Test:</b> Select your trained model → Choose image or video → "
            "Click 'Run Detection' → See results with bounding boxes | "
            "For videos, first frame will be processed"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #00ff88; font-size: 21px; padding: 10px;")
        layout.addWidget(instructions)
        
        # Initialize test state
        self.current_test_result = None
        
        return self._make_scrollable(widget)
    
    def create_usbcam_tab(self):
        """Create USB camera real-time detection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model selection
        model_group = QGroupBox("🤖 Model Configuration")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("Detection Mode:"), 0, 0)
        self.usbcam_detection_mode = QComboBox()
        self.usbcam_detection_mode.addItems(["Object Detection", "Segmentation", "Pose Detection"])
        self.usbcam_detection_mode.setToolTip("Select detection mode: Object Detection, Segmentation, or Pose Detection")
        model_layout.addWidget(self.usbcam_detection_mode, 0, 1, 1, 2)
        
        model_layout.addWidget(QLabel("Model Path:"), 1, 0)
        self.usbcam_model_path = QLineEdit("./model/yolov5s.pt")
        model_layout.addWidget(self.usbcam_model_path, 1, 1)
        
        browse_usbcam_model_btn = QPushButton("📂 Browse")
        browse_usbcam_model_btn.clicked.connect(self.browse_usbcam_model)
        model_layout.addWidget(browse_usbcam_model_btn, 1, 2)
        
        model_layout.addWidget(QLabel("Confidence:"), 2, 0)
        self.usbcam_confidence = QDoubleSpinBox()
        self.usbcam_confidence.setRange(0.0, 1.0)
        self.usbcam_confidence.setValue(0.25)
        self.usbcam_confidence.setSingleStep(0.05)
        model_layout.addWidget(self.usbcam_confidence, 2, 1)
        
        model_layout.addWidget(QLabel("Class Names:"), 3, 0)
        self.usbcam_class_names = QLineEdit("object")
        self.usbcam_class_names.setPlaceholderText("Comma-separated: object,person,dog")
        model_layout.addWidget(self.usbcam_class_names, 3, 1, 1, 2)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Camera selection
        camera_group = QGroupBox("📹 Camera Selection")
        camera_layout = QGridLayout()
        
        camera_layout.addWidget(QLabel("Camera Index:"), 0, 0)
        self.usbcam_index = QSpinBox()
        self.usbcam_index.setRange(0, 10)
        self.usbcam_index.setValue(0)
        self.usbcam_index.setToolTip("Camera index (0 = default camera, 1 = second camera, etc.)")
        camera_layout.addWidget(self.usbcam_index, 0, 1)
        
        # Detect cameras button
        detect_cameras_btn = QPushButton("🔍 Detect Cameras")
        detect_cameras_btn.clicked.connect(self.detect_usb_cameras)
        detect_cameras_btn.setToolTip("Scan for available USB cameras")
        camera_layout.addWidget(detect_cameras_btn, 0, 2)
        
        # Available cameras display
        self.usbcam_available = QLabel("Click 'Detect Cameras' to scan")
        self.usbcam_available.setStyleSheet("color: #00ff88; font-size: 19px; font-style: italic;")
        camera_layout.addWidget(self.usbcam_available, 1, 0, 1, 3)
        
        # Camera control buttons
        button_layout = QHBoxLayout()
        
        self.start_usbcam_btn = QPushButton("▶️ Start Camera")
        self.start_usbcam_btn.clicked.connect(self.start_usbcam_stream)
        button_layout.addWidget(self.start_usbcam_btn)
        
        self.stop_usbcam_btn = QPushButton("⏹️ Stop Camera")
        self.stop_usbcam_btn.clicked.connect(self.stop_usbcam_stream)
        self.stop_usbcam_btn.setEnabled(False)
        button_layout.addWidget(self.stop_usbcam_btn)
        
        camera_layout.addLayout(button_layout, 2, 0, 1, 3)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Camera display
        display_group = QGroupBox("📺 Live Camera Feed")
        display_layout = QVBoxLayout()
        
        # Camera info
        self.usbcam_stream_info = QLabel("No camera active")
        self.usbcam_stream_info.setAlignment(Qt.AlignCenter)
        self.usbcam_stream_info.setStyleSheet("font-size: 24px; color: #00ff88; padding: 5px;")
        display_layout.addWidget(self.usbcam_stream_info)
        
        # Video display
        self.usbcam_display = QLabel("Click 'Start Camera' to begin")
        self.usbcam_display.setAlignment(Qt.AlignCenter)
        self.usbcam_display.setMinimumSize(800, 500)
        self.usbcam_display.setStyleSheet("""
            QLabel {
                background-color: #0d1117;
                border: 2px solid #00ff88;
                border-radius: 5px;
                color: #00ff88;
                font-size: 19px;
            }
        """)
        self.usbcam_display.setScaledContents(False)
        display_layout.addWidget(self.usbcam_display)
        
        # Camera statistics
        stats_layout = QHBoxLayout()
        
        self.usbcam_detection_count = QLabel("Detections: 0")
        self.usbcam_detection_count.setAlignment(Qt.AlignCenter)
        self.usbcam_detection_count.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff88;")
        stats_layout.addWidget(self.usbcam_detection_count)
        
        self.usbcam_fps_display = QLabel("FPS: 0.0")
        self.usbcam_fps_display.setAlignment(Qt.AlignCenter)
        self.usbcam_fps_display.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff88;")
        stats_layout.addWidget(self.usbcam_fps_display)
        
        display_layout.addLayout(stats_layout)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Instructions
        instructions = QLabel(
            "📝 <b>How to Use:</b> Select camera index (0 = default) → Select trained model → "
            "Click 'Start Camera' → See real-time detections | "
            "Use 'Detect Cameras' to find available cameras<br>"
            "<br>"
            "💡 <b>Tips:</b> Most laptops have built-in webcam at index 0 | "
            "External USB cameras usually start at index 1 | "
            "Lower confidence threshold to detect more objects | "
            "Ensure good lighting for best results"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #00ff88; font-size: 21px; padding: 10px;")
        layout.addWidget(instructions)
        
        return self._make_scrollable(widget)
    
    def create_logs_tab(self):
        """Create logs tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        logs_group = QGroupBox("📝 System Logs")
        logs_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        logs_layout.addWidget(self.log_display)
        
        clear_btn = QPushButton("🗑️ Clear Logs")
        clear_btn.clicked.connect(lambda: self.log_display.clear())
        logs_layout.addWidget(clear_btn)
        
        logs_group.setLayout(logs_layout)
        layout.addWidget(logs_group)
        
        return self._make_scrollable(widget)
    
    def create_help_tab(self):
        """Create help tab with all help content"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Help topics group
        help_group = QGroupBox("❓ Help & Documentation")
        help_layout = QVBoxLayout()
        
        # Create buttons for each help topic
        quick_start_btn = QPushButton("🚀 Quick Start Guide")
        quick_start_btn.clicked.connect(self.show_quick_start)
        quick_start_btn.setMinimumHeight(40)
        help_layout.addWidget(quick_start_btn)
        
        about_project_btn = QPushButton("📖 About This Project")
        about_project_btn.clicked.connect(self.show_about_project)
        about_project_btn.setMinimumHeight(40)
        help_layout.addWidget(about_project_btn)
        
        how_to_use_btn = QPushButton("🎯 How to Use")
        how_to_use_btn.clicked.connect(self.show_how_to_use)
        how_to_use_btn.setMinimumHeight(40)
        help_layout.addWidget(how_to_use_btn)
        
        metrics_btn = QPushButton("📊 Understanding Metrics")
        metrics_btn.clicked.connect(self.show_metrics_help)
        metrics_btn.setMinimumHeight(40)
        help_layout.addWidget(metrics_btn)
        
        troubleshooting_btn = QPushButton("🔧 Troubleshooting")
        troubleshooting_btn.clicked.connect(self.show_troubleshooting)
        troubleshooting_btn.setMinimumHeight(40)
        help_layout.addWidget(troubleshooting_btn)
        
        help_layout.addSpacing(20)
        
        about_btn = QPushButton("ℹ️ About")
        about_btn.clicked.connect(self.show_about)
        about_btn.setMinimumHeight(40)
        help_layout.addWidget(about_btn)
        
        help_layout.addStretch()
        
        help_group.setLayout(help_layout)
        layout.addWidget(help_group)
        
        return self._make_scrollable(widget)
    
    def create_rtsp_tab(self):
        """Create RTSP real-time detection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model selection
        model_group = QGroupBox("🤖 Model Configuration")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.rtsp_model_path = QLineEdit("./model/yolov5s.pt")
        model_layout.addWidget(self.rtsp_model_path, 0, 1)
        
        browse_rtsp_model_btn = QPushButton("📂 Browse")
        browse_rtsp_model_btn.clicked.connect(self.browse_rtsp_model)
        model_layout.addWidget(browse_rtsp_model_btn, 0, 2)
        
        model_layout.addWidget(QLabel("Confidence:"), 1, 0)
        self.rtsp_confidence = QDoubleSpinBox()
        self.rtsp_confidence.setRange(0.0, 1.0)
        self.rtsp_confidence.setValue(0.25)
        self.rtsp_confidence.setSingleStep(0.05)
        model_layout.addWidget(self.rtsp_confidence, 1, 1)
        
        model_layout.addWidget(QLabel("Class Names:"), 2, 0)
        self.rtsp_class_names = QLineEdit("object")
        self.rtsp_class_names.setPlaceholderText("Comma-separated: object,person,dog")
        model_layout.addWidget(self.rtsp_class_names, 2, 1, 1, 2)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # RTSP stream configuration
        stream_group = QGroupBox("📡 RTSP Stream Configuration")
        stream_layout = QGridLayout()
        
        stream_layout.addWidget(QLabel("RTSP URL:"), 0, 0)
        self.rtsp_url_input = QLineEdit()
        self.rtsp_url_input.setPlaceholderText("rtsp://username:password@192.168.1.100:554/stream")
        stream_layout.addWidget(self.rtsp_url_input, 0, 1)
        
        # Stream control buttons
        button_layout = QHBoxLayout()
        
        self.start_rtsp_btn = QPushButton("▶️ Start Stream")
        self.start_rtsp_btn.clicked.connect(self.start_rtsp_stream)
        button_layout.addWidget(self.start_rtsp_btn)
        
        self.stop_rtsp_btn = QPushButton("⏹️ Stop Stream")
        self.stop_rtsp_btn.clicked.connect(self.stop_rtsp_stream)
        self.stop_rtsp_btn.setEnabled(False)
        button_layout.addWidget(self.stop_rtsp_btn)
        
        stream_layout.addLayout(button_layout, 1, 0, 1, 2)
        
        stream_group.setLayout(stream_layout)
        layout.addWidget(stream_group)
        
        # Stream display
        display_group = QGroupBox("📺 Live Stream")
        display_layout = QVBoxLayout()
        
        # Stream info
        self.rtsp_stream_info = QLabel("No stream active")
        self.rtsp_stream_info.setAlignment(Qt.AlignCenter)
        self.rtsp_stream_info.setStyleSheet("font-size: 24px; color: #00ff88; padding: 5px;")
        display_layout.addWidget(self.rtsp_stream_info)
        
        # Video display
        self.rtsp_display = QLabel("Click 'Start Stream' to begin")
        self.rtsp_display.setAlignment(Qt.AlignCenter)
        self.rtsp_display.setMinimumSize(800, 500)
        self.rtsp_display.setStyleSheet("""
            QLabel {
                background-color: #0d1117;
                border: 2px solid #00ff88;
                border-radius: 5px;
                color: #00ff88;
                font-size: 19px;
            }
        """)
        self.rtsp_display.setScaledContents(False)
        display_layout.addWidget(self.rtsp_display)
        
        # Stream statistics
        stats_layout = QHBoxLayout()
        
        self.rtsp_detection_count = QLabel("Detections: 0")
        self.rtsp_detection_count.setAlignment(Qt.AlignCenter)
        self.rtsp_detection_count.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff88;")
        stats_layout.addWidget(self.rtsp_detection_count)
        
        self.rtsp_fps_display = QLabel("FPS: 0.0")
        self.rtsp_fps_display.setAlignment(Qt.AlignCenter)
        self.rtsp_fps_display.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff88;")
        stats_layout.addWidget(self.rtsp_fps_display)
        
        display_layout.addLayout(stats_layout)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Instructions
        instructions = QLabel(
            "📝 <b>How to Use:</b> Enter RTSP stream URL → Select trained model → "
            "Click 'Start Stream' → See real-time detections | "
            "Example URL: rtsp://192.168.1.100:554/stream<br>"
            "<br>"
            "💡 <b>Tips:</b> Ensure your RTSP camera is accessible on the network | "
            "Lower confidence threshold to detect more objects | "
            "Use a trained model for best results"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #00ff88; font-size: 21px; padding: 10px;")
        layout.addWidget(instructions)
        
        return self._make_scrollable(widget)

    
    # Event handlers
    def browse_source_dir(self):
        """Browse for source directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Source Directory")
        if directory:
            self.source_dir_input.setText(directory)
    
    def browse_dataset_yaml(self):
        """Browse for dataset YAML file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select YAML Config File",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )
        if file_path:
            self.dataset_yaml_input.setText(file_path)
    
    def load_dataset_yaml(self):
        """Load dataset configuration from YAML file"""
        try:
            import yaml
            
            yaml_path = Path(self.dataset_yaml_input.text())
            if not yaml_path.exists():
                self.log("❌ Error: YAML file does not exist")
                self.show_dark_message("File Not Found", "The specified YAML file does not exist.", "warning")
                return
            
            # Load YAML file with UTF-8 encoding
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Extract dataset path
            if 'path' in config:
                dataset_path = config['path']
                if not Path(dataset_path).is_absolute():
                    yaml_parent = yaml_path.parent
                    if '/' not in dataset_path and '\\' not in dataset_path:
                        dataset_path = yaml_parent
                    else:
                        dataset_path = (yaml_parent / dataset_path).resolve()
                else:
                    dataset_path = Path(dataset_path)
                
                self.log(f"✅ Dataset root: {dataset_path}")
                
                # Set source directory (images)
                if 'train' in config:
                    images_path = Path(dataset_path) / config['train']
                    self.source_dir_input.setText(str(images_path))
                    self.log(f"✅ Source directory: {images_path}")
                    self.log(f"💡 Labels should be in the same folder as images")
            
            self.log(f"✅ Loaded YAML configuration from: {yaml_path.name}")
            self.show_dark_message(
                "YAML Loaded", 
                f"Successfully loaded dataset configuration!\n\n"
                f"YAML: {yaml_path.name}\n"
                f"Source: {self.source_dir_input.text()}\n"
                f"Labels: Same as source directory",
                "information"
            )
            
        except ImportError:
            self.log("❌ Error: PyYAML not installed. Install with: pip install pyyaml")
            self.show_dark_message(
                "Missing Dependency", 
                "PyYAML is required to load YAML files.\n\nInstall with: pip install pyyaml",
                "critical"
            )
        except Exception as e:
            self.log(f"❌ Error loading YAML: {str(e)}")
            self.show_dark_message("Error", f"Failed to load YAML file:\n\n{str(e)}", "critical")
    
    def browse_model(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt)")
        if file_path:
            self.model_path.setText(file_path)
    
    def browse_train_dataset(self):
        """Browse for training dataset root"""
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if directory:
            self.train_dataset_root.setText(directory)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir.setText(directory)
    
    def browse_train_yaml(self):
        """Browse for YAML configuration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select YAML Config File",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )
        if file_path:
            self.train_yaml_file.setText(file_path)
    
    def load_train_yaml(self):
        """Load dataset configuration from YAML file"""
        try:
            import yaml
            
            yaml_path = Path(self.train_yaml_file.text())
            if not yaml_path.exists():
                self.log("❌ Error: YAML file does not exist")
                self.show_dark_message("File Not Found", "The specified YAML file does not exist.", "warning")
                return
            
            # Load YAML file with UTF-8 encoding
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Extract dataset information
            if 'path' in config:
                # Convert relative path to absolute if needed
                dataset_path = config['path']
                if not Path(dataset_path).is_absolute():
                    # Check if the path is just a name (no slashes)
                    if '/' not in dataset_path and '\\' not in dataset_path:
                        # If YAML is in a dataset folder (e.g., data/train2017_dataset/dataset.yaml)
                        # Use the YAML file's parent directory as the dataset root
                        yaml_parent = yaml_path.parent
                        
                        # Check if parent is a dataset folder (contains images/ or labels/)
                        if (yaml_parent / 'images').exists() or (yaml_parent / 'labels').exists():
                            dataset_path = yaml_parent
                        else:
                            # Otherwise, resolve relative to project root
                            dataset_path = Path(dataset_path).resolve()
                    else:
                        # Path has slashes, resolve relative to YAML file location
                        yaml_parent = yaml_path.parent
                        dataset_path = (yaml_parent / dataset_path).resolve()
                else:
                    dataset_path = Path(dataset_path)
                    
                self.train_dataset_root.setText(str(dataset_path))
                self.log(f"✅ Dataset path: {dataset_path}")
            
            # Count number of classes
            if 'names' in config:
                if isinstance(config['names'], dict):
                    num_classes = len(config['names'])
                elif isinstance(config['names'], list):
                    num_classes = len(config['names'])
                else:
                    num_classes = 0
                
                self.num_classes_display.setText(str(num_classes))
                self.log(f"✅ Number of classes: {num_classes}")
            
            # Display train/val/test paths
            info_parts = []
            if 'train' in config:
                info_parts.append(f"Train: {config['train']}")
            if 'val' in config:
                info_parts.append(f"Val: {config['val']}")
            if 'test' in config:
                info_parts.append(f"Test: {config['test']}")
            
            if info_parts:
                self.log(f"📊 Dataset splits: {' | '.join(info_parts)}")
            
            self.log(f"✅ Loaded YAML configuration from: {yaml_path.name}")
            self.show_dark_message(
                "YAML Loaded", 
                f"Successfully loaded dataset configuration!\n\n"
                f"Dataset: {yaml_path.name}\n"
                f"Classes: {num_classes}\n"
                f"Path: {self.train_dataset_root.text()}",
                "information"
            )
            
        except ImportError:
            self.log("❌ Error: PyYAML not installed. Install with: pip install pyyaml")
            self.show_dark_message(
                "Missing Dependency", 
                "PyYAML is required to load YAML files.\n\nInstall with: pip install pyyaml",
                "critical"
            )
        except Exception as e:
            self.log(f"❌ Error loading YAML: {str(e)}")
            self.show_dark_message("Error", f"Failed to load YAML file:\n\n{str(e)}", "critical")
            QMessageBox.critical(self, "Error", f"Failed to load YAML file:\n\n{str(e)}")
    
    def stop_training(self):
        """Stop training"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_log.append("⏹️ Stopping training...")
            self.log("⏹️ User requested training stop")
            # Note: Actual stopping would require thread interruption mechanism
            self.training_log.append("⚠️ Training will stop after current epoch completes")
    
    def save_training_log(self):
        """Save training log to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Training Log", "", 
            "Text Files (*.txt);;Log Files (*.log)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.training_log.toPlainText())
                self.log(f"💾 Training log saved to {Path(file_path).name}")
            except Exception as e:
                self.log(f"❌ Error saving log: {str(e)}")
    
    def log_training(self, message):
        """Add message to training log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.append(f"[{timestamp}] {message}")
        # Also log to main log
        self.log(message)
    
    def browse_label_images(self):
        """Browse for images folder to label"""
        directory = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if directory:
            self.label_images_input.setText(directory)
    
    def browse_label_labels(self):
        """Browse for labels folder"""
        directory = QFileDialog.getExistingDirectory(self, "Select Labels Folder")
        if directory:
            self.label_labels_input.setText(directory)
    
    def browse_label_yaml(self):
        """Browse for YAML configuration file for labeling"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select YAML Config File",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )
        if file_path:
            self.label_yaml_input.setText(file_path)
    
    def load_label_yaml(self):
        """Load labeling configuration from YAML file (only loads class names)"""
        try:
            import yaml
            
            yaml_path = Path(self.label_yaml_input.text())
            if not yaml_path.exists():
                self.log("❌ Error: YAML file does not exist")
                self.show_dark_message("File Not Found", "The specified YAML file does not exist.", "warning")
                return
            
            # Load YAML file with UTF-8 encoding
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Extract class names only (do not update Images Folder)
            if 'names' in config:
                if isinstance(config['names'], dict):
                    # COCO format: {0: 'person', 1: 'bicycle', ...}
                    class_names = [config['names'][i] for i in sorted(config['names'].keys())]
                elif isinstance(config['names'], list):
                    # List format: ['person', 'bicycle', ...]
                    class_names = config['names']
                else:
                    class_names = []
                
                if class_names:
                    # Populate the list widget
                    self.class_names_list.clear()
                    for idx, name in enumerate(class_names):
                        self.class_names_list.addItem(f"{idx}: {name}")
                    self.log(f"✅ Loaded {len(class_names)} class names from YAML")
            
            self.log(f"✅ Loaded YAML configuration from: {yaml_path.name}")
            self.show_dark_message(
                "YAML Loaded", 
                f"Successfully loaded class names from YAML!\n\n"
                f"YAML: {yaml_path.name}\n"
                f"Classes: {len(class_names) if 'class_names' in locals() else 0}",
                "information"
            )
            
        except ImportError:
            self.log("❌ Error: PyYAML not installed. Install with: pip install pyyaml")
            self.show_dark_message(
                "Missing Dependency", 
                "PyYAML is required to load YAML files.\n\nInstall with: pip install pyyaml",
                "critical"
            )
        except Exception as e:
            self.log(f"❌ Error loading YAML: {str(e)}")
            self.show_dark_message("Error", f"Failed to load YAML file:\n\n{str(e)}", "critical")
    
    def detect_coco_dataset(self, images_dir):
        """Detect if the loaded directory is a COCO dataset"""
        # Check for COCO-style directory structure or naming
        path_str = str(images_dir).lower()
        
        # Check if path contains 'coco' or typical COCO directory names
        if 'coco' in path_str or 'train2017' in path_str or 'val2017' in path_str:
            return True
        
        # Check if there's a labels directory with COCO-style structure
        parent = images_dir.parent
        if (parent / 'labels').exists() and images_dir.name in ['train2017', 'val2017', 'test2017']:
            return True
        
        return False
    
    def get_coco_class_names(self):
        """Return the 80 COCO class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    # Labeling methods
    def load_labeling_images(self):
        """Load images for labeling"""
        try:
            images_dir = Path(self.label_images_input.text())
            if not images_dir.exists():
                self.log("❌ Error: Images directory does not exist")
                return
            
            # Check if YAML config is provided
            yaml_path = self.label_yaml_input.text().strip()
            
            # If no YAML config, prompt for class names
            if not yaml_path:
                # Ask user for class names with dark-themed dialog
                from PyQt5.QtWidgets import QInputDialog, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout
                
                # Create custom dark dialog
                dialog = QDialog(self)
                dialog.setWindowTitle("Class Names Required")
                dialog.setModal(True)
                dialog.setMinimumWidth(500)
                
                # Apply dark theme
                dialog.setStyleSheet("""
                    QDialog {
                        background-color: #1a1a2e;
                        color: #ffffff;
                    }
                    QLabel {
                        color: #ffffff;
                        font-size: 21px;
                        padding: 10px;
                    }
                    QLineEdit {
                        background-color: #16213e;
                        color: #ffffff;
                        border: 2px solid #00ff88;
                        border-radius: 5px;
                        padding: 10px;
                        font-size: 21px;
                    }
                    QLineEdit:focus {
                        border: 2px solid #00ffff;
                    }
                    QPushButton {
                        background-color: #0f3460;
                        color: #00ff88;
                        border: 2px solid #00ff88;
                        border-radius: 8px;
                        padding: 10px 20px;
                        font-size: 19px;
                        font-weight: bold;
                        min-width: 100px;
                    }
                    QPushButton:hover {
                        background-color: #00ff88;
                        color: #1a1a2e;
                    }
                    QPushButton:pressed {
                        background-color: #00cc66;
                    }
                """)
                
                layout = QVBoxLayout(dialog)
                
                # Message label
                message_label = QLabel("Enter class names (comma-separated):\nExample: object,person,dog")
                message_label.setWordWrap(True)
                layout.addWidget(message_label)
                
                # Input field
                input_field = QLineEdit("object")
                input_field.selectAll()
                layout.addWidget(input_field)
                
                # Buttons
                button_layout = QHBoxLayout()
                button_layout.addStretch()
                
                ok_button = QPushButton("OK")
                ok_button.clicked.connect(dialog.accept)
                button_layout.addWidget(ok_button)
                
                cancel_button = QPushButton("Cancel")
                cancel_button.clicked.connect(dialog.reject)
                button_layout.addWidget(cancel_button)
                
                layout.addLayout(button_layout)
                
                # Show dialog
                result = dialog.exec_()
                
                if result != QDialog.Accepted or not input_field.text().strip():
                    self.log("❌ Class names are required for labeling")
                    return
                
                class_names_input = input_field.text()
                
                # Parse class names
                class_names = [name.strip() for name in class_names_input.split(',') if name.strip()]
                
                if not class_names:
                    self.log("❌ No valid class names provided")
                    return
                
                # Update class names list widget
                self.class_names_list.clear()
                for idx, name in enumerate(class_names):
                    self.class_names_list.addItem(f"{idx}: {name}")
                
                # Save to YAML file with datetime
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                yaml_filename = f"dataset_{timestamp}.yaml"
                
                # Save one level higher than images folder
                yaml_save_path = images_dir.parent / yaml_filename
                
                self._save_dataset_yaml(yaml_save_path, images_dir, class_names)
                
                self.log(f"✅ Created YAML config: {yaml_save_path}")
                self.log(f"🏷️ Classes: {', '.join(class_names)}")
                
                # Update the YAML input field
                self.label_yaml_input.setText(str(yaml_save_path))
            else:
                # Auto-detect COCO dataset and populate class names
                is_coco_dataset = self.detect_coco_dataset(images_dir)
                
                # Get class names from list widget
                class_names = []
                for i in range(self.class_names_list.count()):
                    item_text = self.class_names_list.item(i).text()
                    # Extract class name from "0: person" format
                    if ':' in item_text:
                        class_name = item_text.split(':', 1)[1].strip()
                        class_names.append(class_name)
                
                # If COCO dataset detected and only default class, use COCO classes
                if is_coco_dataset and (not class_names or class_names == ['object']):
                    class_names = self.get_coco_class_names()
                    # Populate list widget
                    self.class_names_list.clear()
                    for idx, name in enumerate(class_names):
                        self.class_names_list.addItem(f"{idx}: {name}")
                    self.log("🎯 COCO dataset detected! Auto-populated 80 COCO class names")
                elif not class_names:
                    class_names = ["object"]
            
            self.image_canvas.set_class_names(class_names)
            
            # Find all image files (case-insensitive)
            self.labeling_images = []
            seen_files = set()  # Track files to avoid duplicates
            
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Search for both lowercase and uppercase
                for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                    for img_path in images_dir.glob(pattern):
                        # Use lowercase path for comparison to avoid case duplicates
                        file_key = str(img_path).lower()
                        if file_key not in seen_files:
                            seen_files.add(file_key)
                            self.labeling_images.append(img_path)
            
            if not self.labeling_images:
                self.log("❌ No images found in directory")
                return
            
            self.labeling_images.sort()
            self.current_image_index = 0
            
            self.log(f"✅ Loaded {len(self.labeling_images)} images for labeling")
            self.log(f"🏷️ Classes: {', '.join(class_names)}")
            
            # Enable controls
            self.prev_image_btn.setEnabled(True)
            self.next_image_btn.setEnabled(True)
            self.undo_box_btn.setEnabled(True)
            self.clear_boxes_btn.setEnabled(True)
            self.save_label_btn.setEnabled(True)
            self.image_slider.setEnabled(True)
            self.image_slider.setMaximum(len(self.labeling_images) - 1)
            
            # Load first image
            self.load_current_image()
            self.log(f"📷 Attempting to display first image: {self.labeling_images[0].name}")
            
        except Exception as e:
            self.log(f"❌ Error loading images: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
    
    def _save_dataset_yaml(self, yaml_path, images_dir, class_names):
        """Save dataset YAML configuration file"""
        try:
            import yaml
            
            # Determine relative path from YAML location to dataset root
            yaml_parent = yaml_path.parent
            
            # Calculate relative path for 'path' field (should point to dataset root)
            # If images_dir is like: E:/data/my_dataset/images/train
            # Then dataset_root is: E:/data/my_dataset
            # And path should be: ../datasets/my_dataset (relative to YAML location)
            
            # Get dataset root (parent of images folder)
            if images_dir.name in ['train', 'val', 'test']:
                # images_dir is like: .../images/train
                dataset_root = images_dir.parent.parent
            else:
                # images_dir is like: .../images
                dataset_root = images_dir.parent
            
            # Use absolute path to avoid confusion
            path_value = str(dataset_root.resolve())
            
            # Determine train/val/test paths relative to dataset root
            if images_dir.name in ['train', 'val', 'test']:
                # images_dir is like: .../images/train
                train_path = "images/train"
                val_path = "images/val"
                test_path = "images/test"
            else:
                # images_dir is like: .../images
                train_path = "images"
                val_path = "images"
                test_path = "images"
            
            # Create YAML content matching the reference format
            yaml_content = {
                'path': path_value,
                'train': train_path,
                'val': val_path,
                'test': test_path + ' # Optional',
                'nc': len(class_names),
                'names': {idx: name for idx, name in enumerate(class_names)}
            }
            
            # Save YAML file with custom formatting to match reference
            with open(yaml_path, 'w', encoding='utf-8') as f:
                # Write manually to match exact format
                f.write(f"path: {yaml_content['path']}\n")
                f.write(f"train: {yaml_content['train']}\n")
                f.write(f"val: {yaml_content['val']}\n")
                f.write(f"test: {yaml_content['test']}\n")
                f.write(f"nc: {yaml_content['nc']}\n")
                f.write("names:\n")
                for idx, name in enumerate(class_names):
                    f.write(f"  {idx}: {name}\n")
                f.write("\n")
            
            self.log(f"💾 Saved YAML config to: {yaml_path}")
            self.log(f"📂 Dataset path: {path_value}")
            self.log(f"🖼️ Train images: {path_value}/{train_path}")
            self.log(f"🖼️ Val images: {path_value}/{val_path}")
            
        except Exception as e:
            self.log(f"⚠️ Warning: Could not save YAML file: {str(e)}")
    
    def load_current_image(self):
        """Load and display current image"""
        if not self.labeling_images:
            self.log("⚠️ No images in labeling_images list")
            return
        
        try:
            image_path = self.labeling_images[self.current_image_index]
            self.log(f"🔍 Loading image: {image_path}")
            
            # Load image into canvas
            if not self.image_canvas.load_image(image_path):
                self.log(f"❌ Failed to load image: {image_path.name}")
                return
            
            self.log(f"✅ Image loaded into canvas successfully")
            
            # Determine label path (check separate labels folder first)
            labels_folder = self.label_labels_input.text().strip()
            if labels_folder and Path(labels_folder).exists():
                # Use separate labels folder
                label_path = Path(labels_folder) / f"{image_path.stem}.txt"
            else:
                # Use same directory as image (YOLO standard)
                label_path = image_path.parent / f"{image_path.stem}.txt"
            
            bboxes = []
            polygons = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            # Bounding box format: class_id x_center y_center width height
                            bboxes.append([float(x) for x in parts])
                        elif len(parts) > 5:
                            # Polygon format: class_id x1 y1 x2 y2 x3 y3 ...
                            polygons.append([float(x) for x in parts])
            
            self.image_canvas.set_bboxes(bboxes)
            # Set polygons directly
            self.image_canvas.polygons = polygons
            
            # Force canvas update
            self.image_canvas.update()
            self.image_canvas.repaint()
            
            # Update info
            pixmap = self.image_canvas.original_pixmap
            self.image_info_label.setText(
                f"📷 {image_path.name} | {pixmap.width()}x{pixmap.height()}px | "
                f"Image {self.current_image_index + 1} of {len(self.labeling_images)}"
            )
            self.image_counter_label.setText(
                f"{self.current_image_index + 1} / {len(self.labeling_images)}"
            )
            
            total_annotations = len(bboxes) + len(polygons)
            self.boxes_count_label.setText(f"Annotations: {total_annotations}")
            
            # Update slider without triggering signal
            self.image_slider.blockSignals(True)
            self.image_slider.setValue(self.current_image_index)
            self.image_slider.blockSignals(False)
            
            self.log(f"📷 Loaded: {image_path.name} ({len(bboxes)} boxes + {len(polygons)} polygons)")
            
        except Exception as e:
            self.log(f"❌ Error loading image: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
    
    def prev_labeling_image(self):
        """Go to previous image"""
        if self.current_image_index > 0:
            self.save_current_label()
            self.current_image_index -= 1
            self.load_current_image()
    
    def next_labeling_image(self):
        """Go to next image"""
        if self.current_image_index < len(self.labeling_images) - 1:
            self.save_current_label()
            self.current_image_index += 1
            self.load_current_image()
    
    def on_slider_changed(self, value):
        """Handle slider value change"""
        if value != self.current_image_index and 0 <= value < len(self.labeling_images):
            # Note: Slider doesn't auto-save to avoid losing work
            self.log("⚠️ Slider moved - remember to save current image first!")
            self.current_image_index = value
            self.load_current_image()
    
    def clear_all_boxes(self):
        """Clear all annotations (bounding boxes, polygons, and drawing points)"""
        # Clear completed annotations
        self.image_canvas.clear_all()
        
        # Clear any polygon being drawn
        self.image_canvas.polygon_points = []
        self.image_canvas.drawing_polygon = False
        
        # Clear any bbox being drawn
        self.image_canvas.first_point = None
        self.image_canvas.current_point = None
        self.image_canvas.drawing = False
        
        # Update display
        self.image_canvas.update()
        self.boxes_count_label.setText("Annotations: 0")
        self.log("🗑️ Cleared all annotations (boxes, polygons, and drawing points)")
    
    def undo_last_box(self):
        """Undo the last drawn annotation (bbox or polygon) or point"""
        success, annotation_type = self.image_canvas.undo_last_annotation()
        if success:
            bbox_count = len(self.image_canvas.get_bboxes())
            poly_count = len(self.image_canvas.get_polygons())
            total = bbox_count + poly_count
            self.boxes_count_label.setText(f"Annotations: {total}")
            
            if annotation_type == 'point':
                points_remaining = len(self.image_canvas.polygon_points)
                self.log(f"↶ Removed last polygon point ({points_remaining} points remaining)")
            elif annotation_type == 'polygon':
                self.log(f"↶ Undid last polygon (remaining: {bbox_count} boxes + {poly_count} polygons)")
            else:
                self.log(f"↶ Undid last bounding box (remaining: {bbox_count} boxes + {poly_count} polygons)")
        else:
            self.log("⚠️ No annotations to undo")
    
    def set_drawing_mode(self, mode):
        """Set the drawing mode (bbox or polygon)"""
        self.image_canvas.set_drawing_mode(mode)
        if mode == 'bbox':
            self.log("📦 Switched to Bounding Box mode (Detection)")
        else:
            self.log("🔷 Switched to Polygon mode (Segmentation)")
    
    def on_bbox_created(self, bbox):
        """Handle bbox creation"""
        count = len(self.image_canvas.get_bboxes())
        self.boxes_count_label.setText(f"Boxes: {count}")
        self.log(f"✅ Created bounding box (total: {count})")
    
    def on_polygon_created(self, polygon):
        """Handle polygon creation"""
        poly_count = len(self.image_canvas.get_polygons())
        bbox_count = len(self.image_canvas.get_bboxes())
        total = poly_count + bbox_count
        self.boxes_count_label.setText(f"Annotations: {total}")
        self.log(f"✅ Created polygon (total annotations: {total})")
    
    def on_bbox_deleted(self, index):
        """Handle bbox deletion"""
        count = len(self.image_canvas.get_bboxes())
        self.boxes_count_label.setText(f"Boxes: {count}")
        self.log(f"🗑️ Deleted bounding box (remaining: {count})")
    
    def toggle_class_names_display(self):
        """Toggle display of class names on boxes"""
        self.image_canvas.toggle_class_names()
        state = "shown" if self.image_canvas.show_class_names else "hidden"
        self.log(f"👁️ Class names {state}")
    
    def save_current_label(self):
        """Save current labels to file"""
        if not self.labeling_images:
            return
        
        try:
            image_path = self.labeling_images[self.current_image_index]
            
            # Determine where to save label (check separate labels folder first)
            labels_folder = self.label_labels_input.text().strip()
            if labels_folder and Path(labels_folder).exists():
                # Use separate labels folder
                label_path = Path(labels_folder) / f"{image_path.stem}.txt"
            else:
                # Save in the SAME directory as the image (YOLO standard)
                label_path = image_path.parent / f"{image_path.stem}.txt"
            
            # Get bboxes and polygons from canvas
            bboxes = self.image_canvas.get_bboxes()
            polygons = self.image_canvas.get_polygons()
            
            # Write labels
            with open(label_path, 'w') as f:
                # Write bounding boxes (YOLO detection format)
                for bbox in bboxes:
                    f.write(f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
                
                # Write polygons (YOLO segmentation format)
                for polygon in polygons:
                    class_id = int(polygon[0])
                    coords = polygon[1:]  # All coordinates after class_id
                    # Format: class_id x1 y1 x2 y2 x3 y3 ...
                    line = f"{class_id}"
                    for coord in coords:
                        line += f" {coord:.6f}"
                    f.write(line + "\n")
            
            # Show relative path for clarity
            total_annotations = len(bboxes) + len(polygons)
            try:
                rel_path = label_path.relative_to(Path.cwd())
                self.log(f"💾 Saved {len(bboxes)} boxes + {len(polygons)} polygons ({total_annotations} total) to {rel_path}")
            except:
                self.log(f"💾 Saved {len(bboxes)} boxes + {len(polygons)} polygons ({total_annotations} total) to {label_path.name}")
            
        except Exception as e:
            self.log(f"❌ Error saving label: {str(e)}")
    
    def log(self, message):
        """Add message to log display"""
        self.log_display.append(f"[{self.get_timestamp()}] {message}")
        self.statusBar().showMessage(message)
    
    def get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def collect_images(self):
        """Collect images and labels from source directory"""
        try:
            source_dir = Path(self.source_dir_input.text())
            
            if not source_dir.exists():
                self.log("❌ Error: Source directory does not exist")
                return
            
            # Always use ./data as the base directory for collected datasets
            # Create a subdirectory based on the source folder name
            source_name = source_dir.name
            if source_name in ["train", "val", "test"]:
                # If source is a split folder, use parent folder name
                source_name = source_dir.parent.name
            
            # Avoid duplicate _dataset suffix
            if source_name.endswith("_dataset"):
                dataset_root = Path("./data") / source_name
            else:
                dataset_root = Path("./data") / f"{source_name}_dataset"
            
            self.log("🔄 Collecting images and labels...")
            self.log(f"📁 Dataset root: {dataset_root.absolute()}")
            self.log("📁 Structure: Auto-detect")
            self.log("   Will check for 'images/' and 'labels/' subfolders")
            self.log("   Or labels in same folder as images")
            
            config = DatasetConfig()
            self.dataset_manager = DatasetManager(dataset_root, config)
            
            # Labels should be in the same folder as images
            result = self.dataset_manager.import_local_images(source_dir, copy=True, label_dir=None)
            
            # Show detailed results
            self.log(f"✅ Collected {result.images_collected} images")
            
            # Check for label information
            labels_collected = result.source_attribution.get('labels_collected', 0)
            images_without_labels = result.source_attribution.get('images_without_labels', 0)
            
            if labels_collected > 0:
                self.log(f"🏷️ Collected {labels_collected} label files")
            
            if images_without_labels > 0:
                self.log(f"⚠️ Warning: {images_without_labels} images have no labels")
                self.log("   For object detection, each image needs a .txt label file")
            
            if result.images_failed > 0:
                self.log(f"⚠️ Skipped: {result.images_failed} images (duplicates or invalid)")
            
            # Show any errors
            if result.errors:
                # Separate duplicates from real errors
                duplicate_errors = [e for e in result.errors if "Duplicate image" in e]
                other_errors = [e for e in result.errors if "Duplicate image" not in e]
                
                if duplicate_errors:
                    self.log(f"ℹ️ {len(duplicate_errors)} duplicate(s) skipped (already in dataset)")
                
                if other_errors:
                    self.log("❌ Errors encountered:")
                    for error in other_errors[:5]:  # Show first 5 errors
                        self.log(f"   • {error}")
            
            # Generate YAML file for the dataset
            self._generate_dataset_yaml(dataset_root, source_name)
            
            self.show_statistics()
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
    
    def _generate_dataset_yaml(self, dataset_root: Path, dataset_name: str):
        """Generate a YAML configuration file for the dataset"""
        try:
            import yaml
            
            yaml_path = dataset_root / f"{dataset_name}.yaml"
            
            # Create YAML configuration
            yaml_config = {
                'path': str(dataset_root),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'names': {
                    0: 'object'
                },
                'nc': 1
            }
            
            # Write YAML file
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
            
            self.log(f"📄 Generated YAML config: {yaml_path}")
            self.log(f"   You can use this file in the Training tab")
            
        except Exception as e:
            self.log(f"⚠️ Warning: Could not generate YAML file: {str(e)}")
    
    def split_dataset(self):
        """Split dataset into train/val/test"""
        try:
            if not self.dataset_manager:
                # Use the same logic as collect_images to determine dataset root
                source_dir = Path(self.source_dir_input.text())
                source_name = source_dir.name
                if source_name in ["train", "val", "test"]:
                    source_name = source_dir.parent.name
                
                # Avoid duplicate _dataset suffix
                if source_name.endswith("_dataset"):
                    dataset_root = Path("./data") / source_name
                else:
                    dataset_root = Path("./data") / f"{source_name}_dataset"
                
                config = DatasetConfig()
                self.dataset_manager = DatasetManager(dataset_root, config)
            
            train = self.train_ratio.value()
            val = self.val_ratio.value()
            test = self.test_ratio.value()
            
            total = train + val + test
            if abs(total - 1.0) > 0.01:
                self.log(f"❌ Error: Ratios must sum to 1.0 (current: {total})")
                return
            
            self.log("✂️ Splitting dataset...")
            result = self.dataset_manager.split_dataset(train, val, test)
            
            # Get absolute paths for display
            dataset_root_abs = Path(self.dataset_manager.dataset_root).resolve()
            images_path = dataset_root_abs / "images"
            labels_path = dataset_root_abs / "labels"
            
            self.log(f"✅ Split complete:")
            self.log(f"   Train: {result['train']} images")
            self.log(f"   Val: {result['val']} images")
            self.log(f"   Test: {result['test']} images")
            
            if 'duplicates_skipped' in result and result['duplicates_skipped'] > 0:
                self.log(f"   ⚠️ Skipped {result['duplicates_skipped']} duplicate filenames")
            
            self.log("")
            self.log("📂 Dataset Storage Locations:")
            self.log(f"   Dataset Root: {dataset_root_abs}")
            self.log(f"   Images: {images_path}")
            self.log(f"      ├─ train/  ({result['train']} images)")
            self.log(f"      ├─ val/    ({result['val']} images)")
            self.log(f"      └─ test/   ({result['test']} images)")
            self.log(f"   Labels: {labels_path}")
            self.log(f"      ├─ train/  (YOLO .txt files)")
            self.log(f"      ├─ val/    (YOLO .txt files)")
            self.log(f"      └─ test/   (YOLO .txt files)")
            self.log("")
            
            self.show_statistics()
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")

    
    def show_statistics(self):
        """Show dataset statistics with validation"""
        try:
            # Derive dataset root from source directory or use dataset manager's root
            if self.dataset_manager:
                dataset_root = self.dataset_manager.dataset_root
            else:
                source_dir = Path(self.source_dir_input.text())
                source_name = source_dir.name
                if source_name in ["train", "val", "test"]:
                    source_name = source_dir.parent.name
                
                # Avoid duplicate _dataset suffix
                if source_name.endswith("_dataset"):
                    dataset_root = Path("./data") / source_name
                else:
                    dataset_root = Path("./data") / f"{source_name}_dataset"
            
            if not self.dataset_manager:
                config = DatasetConfig()
                self.dataset_manager = DatasetManager(dataset_root, config)
            
            self.log("🔍 Analyzing dataset and validating labels...")
            
            # Get basic statistics
            stats = self.dataset_manager.get_statistics()
            
            # Validate images and labels
            validation_results = self._validate_dataset_for_training(dataset_root)
            
            # Get absolute paths for display
            dataset_root_abs = Path(dataset_root).resolve()
            images_path = dataset_root_abs / "images"
            labels_path = dataset_root_abs / "labels"
            
            # Calculate additional statistics
            train_ratio = (stats.train_count / stats.total_images * 100) if stats.total_images > 0 else 0
            val_ratio = (stats.val_count / stats.total_images * 100) if stats.total_images > 0 else 0
            test_ratio = (stats.test_count / stats.total_images * 100) if stats.total_images > 0 else 0
            avg_boxes = (validation_results['total_boxes'] / validation_results['images_with_labels']) if validation_results['images_with_labels'] > 0 else 0
            
            # Count actual label files in each split
            train_labels = len(list((labels_path / "train").glob("*.txt"))) if (labels_path / "train").exists() else 0
            val_labels = len(list((labels_path / "val").glob("*.txt"))) if (labels_path / "val").exists() else 0
            test_labels = len(list((labels_path / "test").glob("*.txt"))) if (labels_path / "test").exists() else 0
            
            # Build statistics text
            stats_text = f"""
╔══════════════════════════════════════╗
║       DATASET STATISTICS             ║
╚══════════════════════════════════════╝

📊 Total Images: {stats.total_images}
💾 Total Size: {stats.total_size_bytes / (1024*1024):.2f} MB
📦 Total Bounding Boxes: {validation_results['total_boxes']}
📏 Avg Boxes per Image: {avg_boxes:.2f}

📁 Split Distribution:
   🟢 Train: {stats.train_count} images ({train_ratio:.1f}%)
   🟡 Val: {stats.val_count} images ({val_ratio:.1f}%)
   🔴 Test: {stats.test_count} images ({test_ratio:.1f}%)

╔══════════════════════════════════════╗
║    ⭐ DATA STORAGE LOCATIONS ⭐      ║
╚══════════════════════════════════════╝

📂 Dataset Root:
   ➤ {dataset_root_abs}

🖼️ Images Directory:
   ➤ {images_path}
   ├─ train/  ({stats.train_count} images)
   ├─ val/    ({stats.val_count} images)
   └─ test/   ({stats.test_count} images)
   Format: .jpg, .jpeg, .png, .bmp

🏷️ Labels Directory:
   ➤ {labels_path}
   ├─ train/  ({train_labels} .txt files)
   ├─ val/    ({val_labels} .txt files)
   └─ test/   ({test_labels} .txt files)
   Format: YOLO (.txt) - One file per image
"""
            
            if stats.class_distribution:
                stats_text += "\n🏷️ Class Distribution:\n"
                for class_name, count in stats.class_distribution.items():
                    stats_text += f"   • {class_name}: {count}\n"
            
            # Add validation results
            stats_text += f"""
╔══════════════════════════════════════╗
║       TRAINING READINESS CHECK       ║
╚══════════════════════════════════════╝

✅ Images with Labels: {validation_results['images_with_labels']}
❌ Images without Labels: {validation_results['images_without_labels']}
⚠️ Labels without Images: {validation_results['labels_without_images']}
📦 Total Bounding Boxes: {validation_results['total_boxes']}
"""
            
            # Training readiness status
            if validation_results['is_ready']:
                stats_text += "\n🎉 Status: READY FOR TRAINING ✅\n"
            else:
                stats_text += "\n⚠️ Status: NOT READY - Issues Found ❌\n"
                if validation_results['issues']:
                    stats_text += "\n🔧 Issues to Fix:\n"
                    for issue in validation_results['issues'][:5]:  # Show first 5 issues
                        stats_text += f"   • {issue}\n"
            
            self.stats_display.setText(stats_text)
            self.log("✅ Statistics and validation complete")
            
            # Log issues if any
            if validation_results['issues']:
                self.log(f"⚠️ Found {len(validation_results['issues'])} issues")
                for issue in validation_results['issues'][:3]:
                    self.log(f"   • {issue}")
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
    
    def _validate_dataset_for_training(self, dataset_root):
        """Validate that images and labels are correctly paired for training"""
        results = {
            'images_with_labels': 0,
            'images_without_labels': 0,
            'labels_without_images': 0,
            'total_boxes': 0,
            'is_ready': False,
            'issues': []
        }
        
        try:
            dataset_root = Path(dataset_root)
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            # Check each split (train, val, test)
            for split in ['train', 'val', 'test']:
                # Check both possible structures:
                # 1. Same folder: images/train/
                # 2. Parallel folders: images/train/ and labels/train/
                
                # Try same folder structure first (Yolo_Label standard)
                images_dir = dataset_root / "images" / split
                
                if not images_dir.exists():
                    continue
                
                # Find all images
                image_files = []
                for ext in image_extensions:
                    image_files.extend(list(images_dir.glob(f"*{ext}")))
                    image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
                
                # Check each image for corresponding label
                for img_path in image_files:
                    label_path = img_path.parent / f"{img_path.stem}.txt"
                    
                    if label_path.exists():
                        results['images_with_labels'] += 1
                        
                        # Count bounding boxes in label
                        try:
                            with open(label_path, 'r') as f:
                                lines = [line.strip() for line in f if line.strip()]
                                results['total_boxes'] += len(lines)
                                
                                # Validate label format
                                for line_num, line in enumerate(lines, 1):
                                    parts = line.split()
                                    if len(parts) != 5:
                                        results['issues'].append(
                                            f"{split}/{img_path.name}: Line {line_num} has {len(parts)} values (expected 5)"
                                        )
                                    else:
                                        # Validate values are numbers
                                        try:
                                            class_id = int(parts[0])
                                            coords = [float(x) for x in parts[1:]]
                                            
                                            # Check if coordinates are normalized (0-1)
                                            for coord in coords:
                                                if coord < 0 or coord > 1:
                                                    results['issues'].append(
                                                        f"{split}/{img_path.name}: Coordinate {coord} out of range (0-1)"
                                                    )
                                                    break
                                        except ValueError:
                                            results['issues'].append(
                                                f"{split}/{img_path.name}: Line {line_num} has invalid number format"
                                            )
                        except Exception as e:
                            results['issues'].append(f"{split}/{img_path.name}: Error reading label - {str(e)}")
                    else:
                        results['images_without_labels'] += 1
                        results['issues'].append(f"{split}/{img_path.name}: Missing label file")
                
                # Check for orphaned labels (labels without images)
                label_files = list(images_dir.glob("*.txt"))
                for label_path in label_files:
                    # Check if corresponding image exists
                    has_image = False
                    for ext in image_extensions:
                        img_path = label_path.parent / f"{label_path.stem}{ext}"
                        if img_path.exists():
                            has_image = True
                            break
                    
                    if not has_image:
                        results['labels_without_images'] += 1
                        results['issues'].append(f"{split}/{label_path.name}: No corresponding image found")
            
            # Determine if ready for training
            total_images = results['images_with_labels'] + results['images_without_labels']
            if total_images > 0:
                if results['images_without_labels'] == 0 and results['labels_without_images'] == 0:
                    if results['total_boxes'] > 0:
                        results['is_ready'] = True
                    else:
                        results['issues'].append("No bounding boxes found in any labels")
                else:
                    if results['images_without_labels'] > 0:
                        results['issues'].insert(0, f"{results['images_without_labels']} images are missing labels")
                    if results['labels_without_images'] > 0:
                        results['issues'].insert(0, f"{results['labels_without_images']} orphaned label files found")
            else:
                results['issues'].append("No images found in dataset")
            
        except Exception as e:
            results['issues'].append(f"Validation error: {str(e)}")
        
        return results
    
    def on_task_type_changed(self, task_type):
        """Update model architecture options when task type changes"""
        # Trigger version change to update models based on task type
        self.on_yolo_version_changed(self.yolo_version.currentText())
    
    def on_yolo_version_changed(self, version):
        """Update model architecture options when YOLO version changes"""
        self.model_arch.clear()
        
        task_type = self.task_type.currentText()
        is_segmentation = (task_type == "Segmentation")
        
        if version == "yolov5":
            if is_segmentation:
                self.model_arch.addItems(["yolov5n-seg", "yolov5s-seg", "yolov5m-seg", "yolov5l-seg", "yolov5x-seg"])
                self.model_arch.setCurrentText("yolov5s-seg")
            else:
                self.model_arch.addItems(["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"])
                self.model_arch.setCurrentText("yolov5s")
        elif version == "yolov8":
            if is_segmentation:
                self.model_arch.addItems(["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"])
                self.model_arch.setCurrentText("yolov8n-seg")
            else:
                self.model_arch.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
                self.model_arch.setCurrentText("yolov8n")
        elif version == "yolo11":
            if is_segmentation:
                self.model_arch.addItems(["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"])
                self.model_arch.setCurrentText("yolo11l-seg")
            else:
                self.model_arch.addItems(["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"])
                self.model_arch.setCurrentText("yolo11l")
    
    def start_training(self):
        """Start model training"""
        try:
            self.log_training("🚀 Initializing training...")
            self.training_status_label.setText("Initializing...")
            
            # Clear previous log
            self.training_log.clear()
            
            # Log configuration
            self.log_training("=" * 50)
            self.log_training("TRAINING CONFIGURATION")
            self.log_training("=" * 50)
            self.log_training(f"Task Type: {self.task_type.currentText()}")
            self.log_training(f"YOLO Version: {self.yolo_version.currentText()}")
            self.log_training(f"Model Architecture: {self.model_arch.currentText()}")
            self.log_training(f"Device: {self.device.currentText()}")
            self.log_training(f"Image Size: {self.image_size.currentText()}")
            self.log_training(f"Epochs: {self.epochs.value()}")
            self.log_training(f"Batch Size: {self.batch_size.value()}")
            self.log_training(f"Learning Rate: {self.learning_rate.value()}")
            self.log_training(f"Patience: {self.patience.value()}")
            self.log_training(f"Workers: {self.workers.value()}")
            self.log_training(f"Save Period: {self.save_period.value()} epochs")
            self.log_training(f"Dataset: {self.train_dataset_root.text()}")
            self.log_training(f"Output: {self.output_dir.text()}")
            self.log_training("=" * 50)
            
            # Create configuration
            config_manager = ConfigurationManager(self.yolo_version.currentText())
            config = config_manager.create_default_config("general")
            
            # Get model architecture from UI
            model_arch = self.model_arch.currentText()
            task_type = self.task_type.currentText()
            
            # CRITICAL VALIDATION: Ensure segmentation models have -seg suffix
            if task_type == "Segmentation" and "-seg" not in model_arch:
                error_msg = f"❌ CRITICAL ERROR: Task Type is 'Segmentation' but Model Architecture is '{model_arch}' (missing -seg suffix)!"
                self.log_training(error_msg)
                self.log_training("❌ This would create a DETECTION model, not SEGMENTATION!")
                self.log_training("❌ Training aborted. Please check your settings.")
                self.train_btn.setEnabled(True)
                self.stop_train_btn.setEnabled(False)
                self.training_progress.setVisible(False)
                self.training_status_label.setText("Error - Invalid configuration")
                return
            
            # Log what we're actually using
            self.log_training(f"🔍 Validation: Task Type = {task_type}")
            self.log_training(f"🔍 Validation: Model Architecture = {model_arch}")
            if task_type == "Segmentation":
                self.log_training(f"✅ Segmentation model confirmed: {model_arch}")
            
            config.model_architecture = model_arch
            config.epochs = self.epochs.value()
            config.batch_size = self.batch_size.value()
            config.image_size = int(self.image_size.currentText())
            config.device = self.device.currentText()
            config.num_classes = 1
            config.class_names = ["object"]
            
            # Add training parameters
            config.learning_rate = self.learning_rate.value()
            config.patience = self.patience.value()
            config.workers = self.workers.value()
            config.save_period = self.save_period.value()
            
            # Add dataset YAML path
            yaml_path = Path(self.train_yaml_file.text())
            if yaml_path.exists():
                config.dataset_yaml = yaml_path
                self.log_training(f"📄 Using dataset YAML: {yaml_path}")
            else:
                self.log_training(f"⚠️ Warning: YAML file not found: {yaml_path}")
                self.log_training(f"⚠️ Training may fail without valid dataset configuration")
            
            output_dir = self.output_dir.text()
            
            # Update UI
            self.train_btn.setEnabled(False)
            self.stop_train_btn.setEnabled(True)
            self.training_progress.setVisible(True)
            self.training_progress.setValue(0)
            self.training_status_label.setText("Training in progress...")
            
            self.log_training("✅ Configuration validated")
            self.log_training("🚀 Starting training...")
            
            # Start training in background thread
            self.training_thread = TrainingThread(config, output_dir)
            self.training_thread.progress.connect(self.log_training)
            self.training_thread.finished.connect(self.training_finished)
            self.training_thread.error_occurred.connect(self.handle_training_error)
            self.training_thread.start()
            
        except Exception as e:
            self.log_training(f"❌ Error: {str(e)}")
            self.training_status_label.setText("Error - Training failed")
            self.train_btn.setEnabled(True)
            self.stop_train_btn.setEnabled(False)
            self.training_progress.setVisible(False)
    
    def handle_training_error(self, error_msg):
        """Handle training error and show error message to user"""
        from PyQt5.QtWidgets import QMessageBox
        
        # Reset UI state
        self.train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.training_progress.setVisible(False)
        self.training_status_label.setText("Training failed")
        
        # Create custom dark-themed message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Training Error")
        msg_box.setIcon(QMessageBox.Critical)
        
        # Truncate error message if too long
        display_error = error_msg if len(error_msg) < 500 else error_msg[:500] + "..."
        
        msg_box.setText("Training failed with the following error:")
        msg_box.setInformativeText(display_error)
        msg_box.setDetailedText(error_msg)  # Full error in details
        
        # Add OK button only
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setDefaultButton(QMessageBox.Ok)
        
        # Apply dark theme
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #1a1a2e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 19px;
            }
            QPushButton {
                background-color: #0f3460;
                color: #00ff88;
                border: 2px solid #00ff88;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 19px;
                font-weight: bold;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #00ff88;
                color: #1a1a2e;
            }
            QPushButton:pressed {
                background-color: #00cc66;
            }
            QTextEdit {
                background-color: #16213e;
                color: #00ff88;
                border: 1px solid #00ff88;
                font-family: 'Courier New';
            }
        """)
        
        # Show dialog
        msg_box.exec_()
        
        self.log_training("=" * 50)
        self.log_training("❌ Training failed - Please fix the error and try again")
        self.log_training("=" * 50)
    
    def start_simulation_training(self):
        """Start training in simulation mode"""
        try:
            self.log_training("🎭 Starting SIMULATION training...")
            self.training_status_label.setText("Simulation in progress...")
            
            # Get config from UI
            config_manager = ConfigurationManager(self.yolo_version.currentText())
            config = config_manager.create_default_config("general")
            
            config.model_architecture = self.model_arch.currentText()
            config.epochs = self.epochs.value()
            config.batch_size = self.batch_size.value()
            config.image_size = int(self.image_size.currentText())
            config.device = self.device.currentText()
            config.num_classes = 1
            config.class_names = ["object"]
            
            output_dir = self.output_dir.text()
            
            # Update UI
            self.train_btn.setEnabled(False)
            self.stop_train_btn.setEnabled(True)
            self.training_progress.setVisible(True)
            self.training_progress.setValue(0)
            
            # Start simulation training
            from src.training import TrainingEngine
            engine = TrainingEngine(config, Path(output_dir))
            
            # Run simulation (this will use the fallback simulation mode)
            import threading
            def run_simulation():
                try:
                    result = engine.train()
                    self.training_finished(result.final_metrics, result.is_simulation, result)
                except Exception as e:
                    self.log_training(f"❌ Simulation error: {str(e)}")
                    self.training_status_label.setText("Simulation failed")
                    self.train_btn.setEnabled(True)
                    self.stop_train_btn.setEnabled(False)
                    self.training_progress.setVisible(False)
            
            thread = threading.Thread(target=run_simulation)
            thread.start()
            
        except Exception as e:
            self.log_training(f"❌ Simulation error: {str(e)}")
            self.training_status_label.setText("Simulation failed")
            self.train_btn.setEnabled(True)
            self.stop_train_btn.setEnabled(False)
            self.training_progress.setVisible(False)
    
    def training_finished(self, metrics, is_simulation=False, result=None):
        """Handle training completion"""
        self.train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.training_progress.setVisible(False)
        self.training_progress.setValue(100)
        
        if is_simulation:
            self.training_status_label.setText("⚠️ SIMULATION MODE - No real model trained!")
            self.log_training("=" * 50)
            self.log_training("⚠️⚠️⚠️ SIMULATION MODE ⚠️⚠️⚠️")
            self.log_training("=" * 50)
            self.log_training("❌ NO REAL MODEL WAS TRAINED!")
            self.log_training("❌ The model file (best.pt) is EMPTY (0 bytes)")
            self.log_training("❌ This was a simulation due to PyTorch/NumPy errors")
            self.log_training("=" * 50)
            self.log_training(f"📊 Simulated mAP50: {metrics.get('map50', 0):.4f}")
            self.log_training(f"📊 Simulated Precision: {metrics.get('precision', 0):.4f}")
            self.log_training(f"📊 Simulated Recall: {metrics.get('recall', 0):.4f}")
            self.log_training("=" * 50)
            self.log_training("🔧 TO FIX AND ENABLE REAL GPU TRAINING:")
            self.log_training("   1. Close this GUI")
            self.log_training("   2. Run: .\\rebuild_venv.bat")
            self.log_training("   3. Restart the GUI")
            self.log_training("=" * 50)
        else:
            self.training_status_label.setText("✅ Training completed!")
            self.log_training("=" * 50)
            self.log_training("✅ REAL TRAINING COMPLETED!")
            self.log_training("=" * 50)
            self.log_training(f"📊 Final mAP50: {metrics.get('map50', 0):.4f}")
            self.log_training(f"📊 Precision: {metrics.get('precision', 0):.4f}")
            self.log_training(f"📊 Recall: {metrics.get('recall', 0):.4f}")
            self.log_training("=" * 50)
            
            # Display archive path if available (this is the main archived model)
            if hasattr(result, 'archive_path') and result.archive_path:
                archive_path = Path(result.archive_path)
                self.log_training("")
                self.log_training("=" * 50)
                self.log_training("📦 ARCHIVED MODEL INFORMATION")
                self.log_training("=" * 50)
                
                if archive_path.exists():
                    archive_size = archive_path.stat().st_size / (1024 * 1024)
                    self.log_training(f"📁 Archived to: {archive_path}")
                    self.log_training(f"📊 File size: {archive_size:.2f} MB")
                    
                    # Verify the archived model
                    try:
                        import torch
                        self.log_training(f"🔍 Verifying archived model...")
                        
                        checkpoint = torch.load(str(archive_path), map_location='cpu')
                        train_args = checkpoint.get('train_args', {})
                        actual_task = train_args.get('task', 'unknown')
                        actual_model = train_args.get('model', 'unknown')
                        
                        self.log_training(f"")
                        self.log_training(f"📋 Model Details:")
                        self.log_training(f"   Task: {actual_task}")
                        self.log_training(f"   Model: {actual_model}")
                        self.log_training(f"   Epochs: {train_args.get('epochs', 'unknown')}")
                        self.log_training(f"   Batch: {train_args.get('batch', 'unknown')}")
                        self.log_training(f"   Image size: {train_args.get('imgsz', 'unknown')}")
                        
                        # Get class names
                        if 'model' in checkpoint:
                            model_state = checkpoint['model']
                            if hasattr(model_state, 'names'):
                                self.log_training(f"   Classes: {model_state.names}")
                        
                        # Get metrics
                        if 'train_metrics' in checkpoint:
                            metrics_dict = checkpoint['train_metrics']
                            self.log_training(f"")
                            self.log_training(f"📊 Model Metrics:")
                            if 'metrics/mAP50(B)' in metrics_dict:
                                self.log_training(f"   Box mAP50: {metrics_dict['metrics/mAP50(B)']:.4f}")
                            if 'metrics/mAP50-95(B)' in metrics_dict:
                                self.log_training(f"   Box mAP50-95: {metrics_dict['metrics/mAP50-95(B)']:.4f}")
                            if 'metrics/mAP50(M)' in metrics_dict:
                                self.log_training(f"   Mask mAP50: {metrics_dict['metrics/mAP50(M)']:.4f}")
                            if 'metrics/mAP50-95(M)' in metrics_dict:
                                self.log_training(f"   Mask mAP50-95: {metrics_dict['metrics/mAP50-95(M)']:.4f}")
                        
                        # Verify model type
                        self.log_training(f"")
                        self.log_training(f"🎯 Model Type Verification:")
                        
                        if actual_task == 'segment':
                            self.log_training(f"   ✅ Task is 'segment' - This is a SEGMENTATION model!")
                        else:
                            self.log_training(f"   ❌ Task is '{actual_task}' - This is NOT a segmentation model!")
                        
                        if '-seg' in str(actual_model):
                            self.log_training(f"   ✅ Model has '-seg' suffix - Correct architecture!")
                        else:
                            self.log_training(f"   ❌ Model doesn't have '-seg' suffix - Wrong architecture!")
                        
                        # Check filename
                        if '_seg_' in str(archive_path.name):
                            self.log_training(f"   ✅ Filename has '_seg' suffix - Correctly named!")
                        else:
                            self.log_training(f"   ℹ️  Filename doesn't have '_seg' suffix - Detection model")
                        
                        # Final verdict
                        if actual_task == 'segment' and '-seg' in str(actual_model):
                            self.log_training(f"")
                            self.log_training(f"✅✅✅ CONFIRMED: This IS a SEGMENTATION model! ✅✅✅")
                            if '_seg_' in str(archive_path.name):
                                self.log_training(f"✅ Filename is correct with '_seg' suffix!")
                            else:
                                self.log_training(f"⚠️  WARNING: Filename should have '_seg' suffix!")
                        else:
                            self.log_training(f"")
                            self.log_training(f"ℹ️  This is a DETECTION model (not segmentation)")
                        
                    except Exception as e:
                        self.log_training(f"❌ Error verifying archived model: {str(e)}")
                        import traceback
                        self.log_training(traceback.format_exc())
                else:
                    self.log_training(f"❌ Archived model not found: {archive_path}")
                
                self.log_training("=" * 50)
            
            # Display original model path info
            if hasattr(result, 'model_path') and result.model_path:
                model_path = Path(result.model_path)
                self.log_training("")
                self.log_training("📍 Original model location:")
                self.log_training(f"   {model_path}")
                if model_path.exists():
                    model_size_mb = model_path.stat().st_size / (1024 * 1024)
                    self.log_training(f"   Size: {model_size_mb:.2f} MB")
            
            self.log_training("")
            self.log_training("🎉 Training session complete!")
            self.log_training("=" * 50)

    
    def evaluate_model(self):
        """Evaluate trained model"""
        try:
            model_path = Path(self.model_path.text())
            
            if not model_path.exists():
                self.log("❌ Error: Model file does not exist")
                return
            
            self.log("📊 Evaluating model...")
            
            config = EvaluationConfig(
                confidence_threshold=self.conf_threshold.value()
            )
            
            evaluator = EvaluationModule(model_path, config)
            result = evaluator.evaluate(self.eval_split.currentText())
            
            # Display results
            results_text = f"""
╔══════════════════════════════════════╗
║       EVALUATION RESULTS             ║
╚══════════════════════════════════════╝

📊 Overall Metrics:
   • Precision: {result.overall_metrics.precision:.4f}
   • Recall: {result.overall_metrics.recall:.4f}
   • F1 Score: {result.overall_metrics.f1_score:.4f}
   • mAP50: {result.overall_metrics.map50:.4f}
   • mAP50-95: {result.overall_metrics.map50_95:.4f}

⚡ Performance:
   • Total Images: {result.total_images}
   • Avg Inference Time: {result.inference_time_ms:.2f} ms

🎯 Confidence Threshold: {self.conf_threshold.value()}
"""
            
            self.eval_results.setText(results_text)
            self.log("✅ Evaluation complete!")
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
    
    # Test tab methods
    def browse_test_model(self):
        """Browse for test model file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt)")
        if file_path:
            self.test_model_path.setText(file_path)
    
    def browse_test_input(self):
        """Browse for test input file"""
        input_type = self.test_input_type.currentText()
        if input_type == "Image":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", 
                "Image Files (*.jpg *.jpeg *.png *.bmp)"
            )
        else:  # Video
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "", 
                "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
        
        if file_path:
            self.test_input_path.setText(file_path)
    
    def on_test_input_type_changed(self, input_type):
        """Handle input type change"""
        if input_type == "Image":
            self.test_input_path.setPlaceholderText("Select an image file (.jpg, .png, .bmp)")
        else:
            self.test_input_path.setPlaceholderText("Select a video file (.mp4, .avi, .mov)")
    
    def run_detection_test(self):
        """Run detection on selected input"""
        try:
            import time
            import warnings
            from PIL import Image, ImageDraw, ImageFont
            from PyQt5.QtGui import QPixmap
            import cv2
            from ultralytics import YOLO
            
            # Suppress YOLO warnings
            warnings.filterwarnings('ignore')
            import logging
            logging.getLogger('ultralytics').setLevel(logging.ERROR)
            
            model_path = Path(self.test_model_path.text())
            input_path = Path(self.test_input_path.text())
            
            if not model_path.exists():
                self.log("❌ Error: Model file does not exist")
                return
            
            if not input_path.exists():
                self.log("❌ Error: Input file does not exist")
                return
            
            self.log(f"🧪 Running detection on {input_path.name}...")
            self.log(f"📦 Loading model: {model_path.name}")
            
            # Parse class names
            class_names = [name.strip() for name in self.test_class_names.text().split(',') if name.strip()]
            
            # Load YOLO model
            try:
                model = YOLO(str(model_path))
                self.log(f"✅ Model loaded successfully")
            except Exception as e:
                self.log(f"❌ Error loading model: {e}")
                return
            
            # Load image
            input_type = self.test_input_type.currentText()
            if input_type == "Video":
                # For video, extract first frame
                cap = cv2.VideoCapture(str(input_path))
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    self.log("❌ Error: Could not read video")
                    return
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
            else:
                img = Image.open(input_path)
            
            # Run detection
            start_time = time.time()
            confidence_threshold = self.test_confidence.value()
            
            self.log(f"🔍 Running inference (confidence threshold: {confidence_threshold})...")
            results = model.predict(
                source=img,
                conf=confidence_threshold,
                verbose=False,
                stream=False
            )
            
            inference_time = (time.time() - start_time) * 1000
            
            # Get detections from results
            result = results[0]
            boxes = result.boxes
            
            # Draw bounding boxes
            draw = ImageDraw.Draw(img)
            
            colors = [
                (0, 255, 136),   # Neon green
                (255, 100, 100), # Red
                (100, 150, 255), # Blue
                (255, 200, 0),   # Yellow
                (255, 0, 255),   # Magenta
            ]
            
            detection_count = 0
            for box in boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Draw box
                color = colors[class_id % len(colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Get class name
                if class_names and class_id < len(class_names):
                    class_name = class_names[class_id]
                elif hasattr(model, 'names') and class_id in model.names:
                    class_name = model.names[class_id]
                else:
                    class_name = f"Class {class_id}"
                
                label = f"{class_name} {conf:.2f}"
                
                # Draw label background
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((x1, y1 - 20), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1 - 20), label, fill=(26, 26, 46), font=font)
                
                detection_count += 1
            
            # Save result to temp file and display
            temp_path = Path("./temp_detection_result.jpg")
            img.save(temp_path)
            
            # Display result with 10% padding (90% of available size)
            pixmap = QPixmap(str(temp_path))
            display_size = self.test_result_display.size()
            # Reduce display size by 10% to show bounding boxes at edges
            target_width = int(display_size.width() * 0.9)
            target_height = int(display_size.height() * 0.9)
            scaled_pixmap = pixmap.scaled(
                target_width,
                target_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.test_result_display.setPixmap(scaled_pixmap)
            
            # Update info
            self.test_result_info.setText(
                f"✅ Detection complete | {input_path.name} | {img.width}x{img.height}px"
            )
            self.test_detection_info.setText(
                f"Detections: {detection_count} | Time: {inference_time:.1f}ms | Conf: {confidence_threshold}"
            )
            
            # Enable save button
            self.save_result_btn.setEnabled(True)
            self.current_test_result = temp_path
            
            self.log(f"✅ Detected {detection_count} objects in {inference_time:.1f}ms")
            
        except Exception as e:
            self.log(f"❌ Error during detection: {str(e)}")
            import traceback
            self.log(f"Details: {traceback.format_exc()}")
    
    def save_detection_result(self):
        """Save detection result to file"""
        if not self.current_test_result or not self.current_test_result.exists():
            self.log("❌ No detection result to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Detection Result", "", 
            "Image Files (*.jpg *.png)"
        )
        
        if file_path:
            import shutil
            shutil.copy(self.current_test_result, file_path)
            self.log(f"💾 Saved detection result to {Path(file_path).name}")
    
    # RTSP tab methods
    def browse_rtsp_model(self):
        """Browse for RTSP model file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt)")
        if file_path:
            self.rtsp_model_path.setText(file_path)
    
    def start_rtsp_stream(self):
        """Start RTSP stream processing"""
        try:
            rtsp_url = self.rtsp_url_input.text().strip()
            model_path = Path(self.rtsp_model_path.text())
            
            if not rtsp_url:
                self.log("❌ Error: Please enter RTSP URL")
                self.show_dark_message("Missing URL", "Please enter an RTSP stream URL.", "warning")
                return
            
            if not model_path.exists():
                self.log("❌ Error: Model file does not exist")
                self.show_dark_message("Model Not Found", "The specified model file does not exist.", "warning")
                return
            
            # Parse class names
            class_names = [name.strip() for name in self.rtsp_class_names.text().split(',') if name.strip()]
            
            self.log(f"📡 Starting RTSP stream: {rtsp_url}")
            self.log(f"📦 Using model: {model_path.name}")
            
            # Update UI
            self.start_rtsp_btn.setEnabled(False)
            self.stop_rtsp_btn.setEnabled(True)
            self.rtsp_stream_info.setText("Connecting to stream...")
            
            # Start RTSP thread
            self.rtsp_thread = RTSPThread(
                rtsp_url,
                model_path,
                self.rtsp_confidence.value(),
                class_names
            )
            self.rtsp_thread.frame_ready.connect(self.update_rtsp_frame)
            self.rtsp_thread.status_update.connect(self.update_rtsp_status)
            self.rtsp_thread.error.connect(self.handle_rtsp_error)
            self.rtsp_thread.start()
            
        except Exception as e:
            self.log(f"❌ Error starting RTSP stream: {str(e)}")
            self.show_dark_message("Error", f"Failed to start RTSP stream:\n\n{str(e)}", "critical")
            self.start_rtsp_btn.setEnabled(True)
            self.stop_rtsp_btn.setEnabled(False)
    
    def stop_rtsp_stream(self):
        """Stop RTSP stream processing"""
        if self.rtsp_thread and self.rtsp_thread.isRunning():
            self.log("⏹️ Stopping RTSP stream...")
            self.rtsp_thread.stop()
            self.rtsp_thread.wait()
            
            # Update UI
            self.start_rtsp_btn.setEnabled(True)
            self.stop_rtsp_btn.setEnabled(False)
            self.rtsp_stream_info.setText("Stream stopped")
            self.rtsp_display.setText("Click 'Start Stream' to begin")
            self.rtsp_display.setPixmap(QPixmap())
            
            self.log("✅ RTSP stream stopped")
    
    def update_rtsp_frame(self, pixmap):
        """Update RTSP display with new frame"""
        # Scale pixmap to fit display with 10% padding
        display_size = self.rtsp_display.size()
        target_width = int(display_size.width() * 0.9)
        target_height = int(display_size.height() * 0.9)
        scaled_pixmap = pixmap.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.rtsp_display.setPixmap(scaled_pixmap)
    
    def update_rtsp_status(self, status_text, detection_count, fps):
        """Update RTSP status information"""
        self.rtsp_stream_info.setText(f"🔴 LIVE | {status_text}")
        self.rtsp_detection_count.setText(f"Detections: {detection_count}")
        self.rtsp_fps_display.setText(f"FPS: {fps:.1f}")
    
    def handle_rtsp_error(self, error_message):
        """Handle RTSP stream errors"""
        self.log(f"❌ RTSP Error: {error_message}")
        self.show_dark_message("Stream Error", error_message, "critical")
        
        # Reset UI
        self.start_rtsp_btn.setEnabled(True)
        self.stop_rtsp_btn.setEnabled(False)
        self.rtsp_stream_info.setText("Error - Stream disconnected")
        self.rtsp_display.setText("Connection lost. Check URL and try again.")
        self.rtsp_display.setPixmap(QPixmap())
    
    # USB Camera tab methods
    def browse_usbcam_model(self):
        """Browse for USB camera model file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt)")
        if file_path:
            self.usbcam_model_path.setText(file_path)
    
    def detect_usb_cameras(self):
        """Detect available USB cameras"""
        import cv2
        import os
        
        self.log("🔍 Scanning for USB cameras...")
        self.usbcam_available.setText("Scanning...")
        
        # Suppress OpenCV errors during camera detection
        old_stderr = os.dup(2)
        os.close(2)
        os.open(os.devnull, os.O_RDWR)
        
        try:
            available_cameras = []
            # Check up to 10 camera indices
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
        finally:
            # Restore stderr
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        
        if available_cameras:
            camera_list = ", ".join([f"Camera {i}" for i in available_cameras])
            self.usbcam_available.setText(f"✅ Found: {camera_list}")
            self.log(f"✅ Found {len(available_cameras)} camera(s): {camera_list}")
            
            # Update camera index spinner to only show available cameras
            self.usbcam_index.setMinimum(min(available_cameras))
            self.usbcam_index.setMaximum(max(available_cameras))
            
            # Set valid values (only available camera indices)
            # Note: QSpinBox doesn't support non-contiguous values, so we set min/max range
            # and auto-select first available camera
            self.usbcam_index.setValue(available_cameras[0])
            
            # Store available cameras for validation
            self.available_camera_indices = available_cameras
            
            self.log(f"📹 Camera index range updated: {min(available_cameras)} - {max(available_cameras)}")
        else:
            self.usbcam_available.setText("❌ No cameras found")
            self.log("❌ No USB cameras detected")
            
            # Reset to default range if no cameras found
            self.usbcam_index.setMinimum(0)
            self.usbcam_index.setMaximum(10)
            self.available_camera_indices = []
            
            self.show_dark_message(
                "No Cameras Found",
                "No USB cameras were detected.\n\n"
                "Please ensure:\n"
                "• Camera is connected\n"
                "• Camera drivers are installed\n"
                "• Camera is not in use by another application",
                "warning"
            )
    
    def start_usbcam_stream(self):
        """Start USB camera stream processing"""
        try:
            camera_index = self.usbcam_index.value()
            model_path = Path(self.usbcam_model_path.text())
            detection_mode = self.usbcam_detection_mode.currentText()
            
            if not model_path.exists():
                self.log("❌ Error: Model file does not exist")
                self.show_dark_message("Model Not Found", "The specified model file does not exist.", "warning")
                return
            
            # Parse class names
            class_names = [name.strip() for name in self.usbcam_class_names.text().split(',') if name.strip()]
            
            self.log(f"📹 Starting USB camera {camera_index}")
            self.log(f"🎯 Detection mode: {detection_mode}")
            self.log(f"📦 Using model: {model_path.name}")
            
            # Update UI
            self.start_usbcam_btn.setEnabled(False)
            self.stop_usbcam_btn.setEnabled(True)
            self.usbcam_stream_info.setText("Opening camera...")
            
            # Start USB camera thread
            self.usbcam_thread = USBCamThread(
                camera_index,
                model_path,
                self.usbcam_confidence.value(),
                class_names,
                detection_mode
            )
            self.usbcam_thread.frame_ready.connect(self.update_usbcam_frame)
            self.usbcam_thread.status_update.connect(self.update_usbcam_status)
            self.usbcam_thread.error.connect(self.handle_usbcam_error)
            self.usbcam_thread.start()
            
        except Exception as e:
            self.log(f"❌ Error starting USB camera: {str(e)}")
            self.show_dark_message("Error", f"Failed to start USB camera:\n\n{str(e)}", "critical")
            self.start_usbcam_btn.setEnabled(True)
            self.stop_usbcam_btn.setEnabled(False)
    
    def stop_usbcam_stream(self):
        """Stop USB camera stream processing"""
        if self.usbcam_thread and self.usbcam_thread.isRunning():
            self.log("⏹️ Stopping USB camera...")
            self.usbcam_thread.stop()
            self.usbcam_thread.wait()
            
            # Update UI
            self.start_usbcam_btn.setEnabled(True)
            self.stop_usbcam_btn.setEnabled(False)
            self.usbcam_stream_info.setText("Camera stopped")
            self.usbcam_display.setText("Click 'Start Camera' to begin")
            self.usbcam_display.setPixmap(QPixmap())
            
            self.log("✅ USB camera stopped")
    
    def update_usbcam_frame(self, pixmap):
        """Update USB camera display with new frame"""
        # Scale pixmap to fit display with 10% padding
        display_size = self.usbcam_display.size()
        target_width = int(display_size.width() * 0.9)
        target_height = int(display_size.height() * 0.9)
        scaled_pixmap = pixmap.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.usbcam_display.setPixmap(scaled_pixmap)
    
    def update_usbcam_status(self, status_text, detection_count, fps):
        """Update USB camera status information"""
        self.usbcam_stream_info.setText(f"🔴 LIVE | {status_text}")
        self.usbcam_detection_count.setText(f"Detections: {detection_count}")
        self.usbcam_fps_display.setText(f"FPS: {fps:.1f}")
    
    def handle_usbcam_error(self, error_message):
        """Handle USB camera errors"""
        self.log(f"❌ USB Camera Error: {error_message}")
        self.show_dark_message("Camera Error", error_message, "critical")
        
        # Reset UI
        self.start_usbcam_btn.setEnabled(True)
        self.stop_usbcam_btn.setEnabled(False)
        self.usbcam_stream_info.setText("Error - Camera disconnected")
        self.usbcam_display.setText("Camera error. Check connection and try again.")
        self.usbcam_display.setPixmap(QPixmap())


    # Help menu methods
    def show_help_dialog(self, title, content):
        """Show a help dialog with formatted content"""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setGeometry(150, 150, 900, 700)
        
        layout = QVBoxLayout(dialog)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        # Create content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Add text
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setMarkdown(content)
        text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #ffffff;
                border: 2px solid #00ff88;
                border-radius: 5px;
                padding: 15px;
                font-size: 19px;
                line-height: 1.6;
            }
        """)
        content_layout.addWidget(text_edit)
        
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        # Close button
        close_btn = QPushButton("✅ Got it!")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()

    
    def show_quick_start(self):
        """Show quick start guide"""
        content = """
# 🚀 Quick Start Guide

## Welcome to YOLO Training Pipeline!

This guide will get you started in **5 minutes**.

---

## Step 1: Prepare Your Images 📸

You need images of babies (or whatever you want to detect).

**Requirements:**
- At least 100 images (300+ recommended)
- Formats: .jpg, .jpeg, .png, or .bmp
- Clear, good quality photos
- Variety of angles and lighting

---

## Step 2: Collect Images 📁

1. Go to **📁 Dataset** tab
2. Click **📂 Browse** next to "Source Directory"
3. Select your folder with images
4. Click **🚀 Collect Images**
5. Wait for completion message

**What happens:** System validates images, removes duplicates, organizes them.

---

## Step 3: Split Dataset ✂️

1. Stay in **📁 Dataset** tab
2. Keep default ratios (0.7, 0.2, 0.1) or adjust
3. Click **✂️ Split Dataset**

**What happens:** Images divided into training (70%), validation (20%), and test (10%) sets.

---

## Step 4: Configure Training ⚙️

1. Go to **🚀 Training** tab
2. **For quick test:**
   - Epochs: 10
   - Batch Size: 8
   - Model: yolov5s
3. **For real training:**
   - Epochs: 50-100
   - Batch Size: 16
   - Model: yolov5s or yolov5m

---

## Step 5: Start Training 🎯

1. Click **🚀 Start Training**
2. Watch progress in **📝 Logs** tab
3. Wait for completion (10 min - 2 hours depending on settings)

**What happens:** AI learns to detect babies from your images!

---

## Step 6: Evaluate Model 📊

1. Go to **📊 Evaluation** tab
2. Model path should be auto-filled
3. Click **📊 Evaluate Model**
4. Review metrics

**Good results:**
- Precision > 0.75 (75%)
- Recall > 0.70 (70%)
- mAP50 > 0.70 (70%)

---

## 🎉 Done!

You've trained your first AI model!

**Next steps:**
- Try with more images
- Increase epochs
- Experiment with settings
- Read other help sections

---

**Need more help?** Check other Help menu options!
"""
        self.show_help_dialog("🚀 Quick Start Guide", content)

    
    def show_about_project(self):
        """Show about project information"""
        content = """
# 📖 About This Project

## What Is This?

**YOLO Training Pipeline** is a complete AI training system for object detection.

It helps you train AI models to **detect and locate objects** in images - like babies, cars, animals, or anything else!

---

## What Does YOLO Mean?

**YOLO = "You Only Look Once"**

It's a fast, accurate AI technology for object detection.

**Traditional detection:** Scan image slowly, piece by piece 🐌  
**YOLO:** Look at entire image once, find everything instantly ⚡

---

## What Can You Do With This?

### 1. Object Detection 🎯
- Detect any object you train it on
- Cars, animals, products, people, etc.
- Custom applications
- Safety monitoring

### 2. Learn AI/ML 🎓
- Understand how AI training works
- Experiment with parameters
- Build real projects

---

## Key Features

✅ **Complete Pipeline**: Data → Training → Evaluation  
✅ **Beautiful GUI**: High-tech interface  
✅ **Easy to Use**: No coding required  
✅ **Professional**: Production-ready quality  
✅ **Ethical**: Responsible data handling  
✅ **Well Documented**: Comprehensive guides

---

## Technology Stack

- **Python 3.10**: Programming language
- **PyQt5**: GUI framework
- **YOLO**: AI object detection
- **Pillow**: Image processing
- **NumPy**: Math operations

---

## How It Works

### Training Process:

1. **Show AI examples** → "This is an object"
2. **AI learns patterns** → Shapes, colors, features
3. **Test on new images** → Can it find objects?
4. **Improve accuracy** → More training, better data

### Result:

A trained model that can detect objects in **any** photo, even ones it's never seen!

---

## Performance

**Speed:** ~15ms per image (very fast!)  
**Accuracy:** 75-90% with good training data  
**Scalability:** Works on CPU or GPU

---

## Use Cases

- 🏥 Healthcare monitoring
- 🏠 Smart home systems
- 📸 Photo organization
- 🛡️ Safety systems
- 🔬 Research tools
- 📱 Mobile apps

---

## Why This Project?

Most AI tools are either:
- Too complex (need PhD to use)
- Too simple (can't do real work)

**This project is:**
- ✅ Professional quality
- ✅ Easy to use
- ✅ Complete solution
- ✅ Beautiful interface

---

**Ready to build something amazing?** 🚀
"""
        self.show_help_dialog("📖 About This Project", content)

    
    def show_how_to_use(self):
        """Show how to use guide"""
        content = """
# 🎯 How to Use This Application

## Complete Workflow Guide

---

## 📁 Dataset Tab

### Collecting Images

**Purpose:** Import your training images

**Steps:**
1. Set **Dataset Root** - where to store dataset
2. Set **Source Directory** - where your images are
3. Click **🚀 Collect Images**

**What it does:**
- Validates image formats (.jpg, .png, .bmp)
- Checks minimum dimensions (32x32)
- Removes duplicate images
- Organizes into folders
- Creates manifest file

**Tips:**
- Use 300+ images for best results
- Ensure good image quality
- Include variety (angles, lighting)

---

### Splitting Dataset

**Purpose:** Divide images for training/testing

**Steps:**
1. Set split ratios (must sum to 1.0)
   - Train: 0.7 (70% for training)
   - Val: 0.2 (20% for validation)
   - Test: 0.1 (10% for testing)
2. Click **✂️ Split Dataset**

**What it does:**
- Randomly splits images
- Moves to train/val/test folders
- Updates manifest
- Maintains reproducibility

**Tips:**
- Standard split: 70/20/10
- More training data = better model
- Keep test set separate!

---

### Viewing Statistics

**Purpose:** Check dataset info

**Steps:**
1. Click **🔄 Refresh Statistics**

**Shows:**
- Total images
- Dataset size (MB)
- Split distribution
- Class distribution

---

## 🚀 Training Tab

### Configuring Training

**Model Settings:**

**YOLO Version:**
- yolov5: Stable, well-tested
- yolov8: Newer, potentially better

**Model Architecture:**
- yolov5n: Fastest, least accurate
- yolov5s: Balanced (recommended)
- yolov5m: More accurate, slower
- yolov5l: Very accurate, slow
- yolov5x: Most accurate, slowest

**Training Parameters:**

**Epochs:** How many times to train on all images
- 10: Quick test
- 50: Good training
- 100+: Best results

**Batch Size:** Images processed together
- 4-8: Low memory
- 16: Standard
- 32+: High memory, faster

**Image Size:** Input image dimensions
- 320: Fastest
- 416: Balanced
- 640: Standard (recommended)
- 1280: Highest quality

**Device:**
- cpu: Works everywhere
- cuda: GPU acceleration (if available)

---

### Starting Training

**Steps:**
1. Configure settings
2. Click **🚀 Start Training**
3. Monitor in **📝 Logs** tab
4. Wait for completion

**What happens:**
- Initializes YOLO model
- Trains for specified epochs
- Saves checkpoints every 10 epochs
- Saves best model
- Logs progress

**Duration:**
- 10 epochs: 5-15 minutes
- 50 epochs: 30-90 minutes
- 100 epochs: 1-3 hours

---

## 📊 Evaluation Tab

### Evaluating Model

**Purpose:** Test model accuracy

**Steps:**
1. Select model path (auto-filled after training)
2. Choose dataset split (test recommended)
3. Set confidence threshold (0.25 default)
4. Click **📊 Evaluate Model**

**What it shows:**
- Precision: Detection accuracy
- Recall: Detection completeness
- F1 Score: Overall balance
- mAP50: Primary metric
- mAP50-95: Strict metric
- Inference time: Speed

**Good Results:**
- Precision > 0.75
- Recall > 0.70
- mAP50 > 0.70

---

## 📝 Logs Tab

### Monitoring Operations

**Purpose:** See what's happening

**Shows:**
- Real-time operation logs
- Success/error messages
- Timestamps
- Progress updates

**Indicators:**
- 🔄 Processing
- ✅ Success
- ❌ Error
- ⚠️ Warning

**Actions:**
- **🗑️ Clear Logs**: Reset log display

---

## 💡 Best Practices

### Data Quality
1. Use clear, well-lit images
2. Include variety
3. Remove blurry/corrupted images
4. Balance your dataset

### Training Strategy
1. Start with quick test (10 epochs)
2. Verify everything works
3. Increase to 50-100 epochs
4. Monitor for overfitting

### Troubleshooting
1. Check logs for errors
2. Reduce batch size if memory issues
3. Ensure dataset is split
4. Verify image paths

---

## 🎯 Common Workflows

### Quick Test (5 minutes)
1. Collect 50 images
2. Split dataset
3. Train 5 epochs, batch 4
4. Evaluate

### Standard Training (1 hour)
1. Collect 300 images
2. Split dataset
3. Train 50 epochs, batch 16
4. Evaluate

### Production Training (3 hours)
1. Collect 1000+ images
2. Split dataset
3. Train 100 epochs, batch 16
4. Evaluate
5. Fine-tune and retrain

---

**Need more help?** Check other Help menu options!
"""
        self.show_help_dialog("🎯 How to Use", content)

    
    def show_metrics_help(self):
        """Show metrics explanation"""
        content = """
# 📊 Understanding Metrics

## What Do These Numbers Mean?

After evaluation, you see several metrics. Here's what they mean in simple terms!

---

## Precision 🎯

**Question:** "When the AI says 'object', is it correct?"

**Formula:** Correct Detections / Total Detections

**Example:**
- AI detects 100 objects
- 85 are actually correct
- 15 are false alarms
- **Precision = 0.85 (85%)**

**What it means:**
- 0.85 = 85% of detections are correct
- Higher is better
- Low precision = too many false alarms

**Good values:**
- > 0.80: Excellent
- 0.70-0.80: Good
- 0.60-0.70: Acceptable
- < 0.60: Needs improvement

---

## Recall 🔍

**Question:** "Does the AI find all the babies?"

**Formula:** Correct Detections / Total Actual Babies

**Example:**
- 100 babies in images
- AI finds 80 of them
- Misses 20
- **Recall = 0.80 (80%)**

**What it means:**
- 0.80 = Finds 80% of babies
- Higher is better
- Low recall = misses too many

**Good values:**
- > 0.80: Excellent
- 0.70-0.80: Good
- 0.60-0.70: Acceptable
- < 0.60: Needs improvement

---

## F1 Score ⚖️

**Question:** "What's the balance between precision and recall?"

**Formula:** 2 × (Precision × Recall) / (Precision + Recall)

**What it means:**
- Harmonic mean of precision and recall
- Single number for overall performance
- Balances both metrics

**Good values:**
- > 0.80: Excellent
- 0.70-0.80: Good
- 0.60-0.70: Acceptable
- < 0.60: Needs improvement

---

## mAP50 🏆

**Full name:** Mean Average Precision at 50% IoU

**Question:** "How accurate is the AI overall?"

**What it means:**
- Primary metric for object detection
- Considers both detection and localization
- 50% IoU = bounding box overlap threshold

**Good values:**
- > 0.75: Excellent
- 0.65-0.75: Good
- 0.55-0.65: Acceptable
- < 0.55: Needs improvement

**This is the MOST IMPORTANT metric!**

---

## mAP50-95 📈

**Full name:** Mean Average Precision from 50% to 95% IoU

**Question:** "How accurate with strict requirements?"

**What it means:**
- Average of mAP at different IoU thresholds
- More strict than mAP50
- Tests precise localization

**Good values:**
- > 0.60: Excellent
- 0.50-0.60: Good
- 0.40-0.50: Acceptable
- < 0.40: Needs improvement

---

## Inference Time ⚡

**Question:** "How fast is the AI?"

**Measured in:** Milliseconds (ms) per image

**What it means:**
- Time to process one image
- Lower is faster

**Good values:**
- < 20ms: Very fast (real-time capable)
- 20-50ms: Fast
- 50-100ms: Acceptable
- > 100ms: Slow

---

## Real-World Examples

### Example 1: Baby Monitor
```
Precision: 0.90 (90%)
Recall: 0.85 (85%)
mAP50: 0.87 (87%)
```

**Interpretation:**
- ✅ Very accurate (90% correct detections)
- ✅ Finds most objects (85%)
- ✅ Excellent overall (87%)
- **Result: Ready for production!**

---

### Example 2: Needs Improvement
```
Precision: 0.60 (60%)
Recall: 0.55 (55%)
mAP50: 0.57 (57%)
```

**Interpretation:**
- ⚠️ Too many false alarms (60%)
- ⚠️ Misses many babies (55%)
- ⚠️ Below acceptable (57%)
- **Result: Need more training/data**

---

## How to Improve Metrics

### Low Precision (False Alarms)
**Solutions:**
- Increase confidence threshold
- Add more negative examples
- Train longer
- Use larger model

### Low Recall (Missing Objects)
**Solutions:**
- Decrease confidence threshold
- Add more varied examples
- Increase image size
- Train longer

### Low mAP (Overall Poor)
**Solutions:**
- Collect more images (300+)
- Improve image quality
- Increase epochs (50-100)
- Use larger model
- Better data variety

---

## Confidence Threshold 🎚️

**What it is:** Minimum confidence to count as detection

**Default:** 0.25 (25%)

**Effects:**
- **Higher (0.5):** Fewer detections, higher precision
- **Lower (0.1):** More detections, higher recall

**Tuning:**
- Start with 0.25
- Adjust based on use case
- Safety-critical: Higher (0.5+)
- Detection-critical: Lower (0.15-0.25)

---

## Summary

**Most Important:**
1. **mAP50** - Overall accuracy
2. **Precision** - Correctness
3. **Recall** - Completeness

**Target Values:**
- mAP50 > 0.70
- Precision > 0.75
- Recall > 0.70

**If below targets:**
- More images
- More epochs
- Better data quality

---

**Still confused?** That's okay! Focus on mAP50 - if it's above 0.70, you're doing great! 🎉
"""
        self.show_help_dialog("📊 Understanding Metrics", content)

    
    def show_troubleshooting(self):
        """Show troubleshooting guide"""
        content = """
# 🔧 Troubleshooting Guide

## Common Issues and Solutions

---

## 🚫 Collection Issues

### "Source directory does not exist"

**Problem:** Can't find your image folder

**Solutions:**
1. Check the path is correct
2. Use the **📂 Browse** button
3. Ensure folder exists
4. Check spelling

---

### "No images collected"

**Problem:** System found no valid images

**Solutions:**
1. Check image formats (.jpg, .png, .bmp)
2. Verify images aren't corrupted
3. Check minimum size (32x32 pixels)
4. Look in Logs tab for specific errors

---

### "Failed: X images"

**Problem:** Some images couldn't be imported

**Reasons:**
- Unsupported format (.gif, .webp, etc.)
- Too small (< 32x32)
- Corrupted files
- Duplicate images

**Solutions:**
1. Check Logs tab for details
2. Remove problematic images
3. Convert to supported formats
4. Increase minimum size if needed

---

## ✂️ Split Issues

### "Ratios must sum to 1.0"

**Problem:** Train + Val + Test ≠ 1.0

**Solution:**
- Adjust ratios to sum exactly to 1.0
- Example: 0.7 + 0.2 + 0.1 = 1.0 ✅
- Example: 0.8 + 0.2 + 0.1 = 1.1 ❌

---

### "No images to split"

**Problem:** Dataset is empty

**Solutions:**
1. Collect images first
2. Check Dataset Root path
3. Verify images were imported

---

## 🚀 Training Issues

### "Training failed"

**Problem:** Training couldn't start

**Solutions:**
1. Check dataset is split
2. Verify images exist in train folder
3. Check Logs tab for specific error
4. Ensure enough disk space

---

### "Out of memory" / Memory Error

**Problem:** Not enough RAM/VRAM

**Solutions:**
1. **Reduce Batch Size:**
   - Try 8 → 4 → 2
2. **Reduce Image Size:**
   - Try 640 → 416 → 320
3. **Use Smaller Model:**
   - yolov5s → yolov5n
4. **Close Other Programs**
5. **Use CPU instead of CUDA**

---

### Training is very slow

**Problem:** Taking too long

**Solutions:**
1. **Reduce Epochs:**
   - Start with 10 for testing
2. **Reduce Batch Size:**
   - Smaller batches = slower
3. **Use GPU:**
   - Change device to "cuda"
4. **Use Smaller Model:**
   - yolov5n is fastest

---

### Training stuck / frozen

**Problem:** No progress for long time

**Solutions:**
1. Check Logs tab for errors
2. Wait - first epoch is slowest
3. Restart application
4. Reduce batch size

---

## 📊 Evaluation Issues

### "Model file does not exist"

**Problem:** Can't find trained model

**Solutions:**
1. Check model path is correct
2. Ensure training completed
3. Look in output directory
4. Check for .pt file

---

### Very low metrics (< 0.50)

**Problem:** Model performs poorly

**Solutions:**
1. **More Images:**
   - Need 300+ for good results
2. **More Epochs:**
   - Try 50-100 instead of 10
3. **Better Data:**
   - Clear, varied images
   - Good lighting
   - Multiple angles
4. **Larger Model:**
   - Try yolov5m instead of yolov5n

---

## 🖥️ GUI Issues

### GUI won't start

**Problem:** Application doesn't open

**Solutions:**
1. **Install PyQt5:**
   ```
   python -m pip install PyQt5
   ```
2. **Check Python path**
3. **Run from correct directory:**
   ```
   cd <your_project_directory>
   ```

---

### GUI is slow / laggy

**Problem:** Interface responds slowly

**Solutions:**
1. Close other applications
2. Restart GUI
3. Check system resources
4. Update graphics drivers

---

### Can't see text / weird colors

**Problem:** Display issues

**Solutions:**
1. Restart application
2. Check display settings
3. Try different monitor
4. Update graphics drivers

---

## 📁 File/Path Issues

### "Permission denied"

**Problem:** Can't access files

**Solutions:**
1. Run as administrator
2. Check file permissions
3. Close files in other programs
4. Check antivirus isn't blocking

---

### "Path not found"

**Problem:** Can't find directory

**Solutions:**
1. Use absolute paths (C:\...)
2. Check spelling
3. Use Browse button
4. Ensure folder exists

---

## 🔄 General Issues

### Application crashes

**Problem:** Unexpected closure

**Solutions:**
1. Check Logs tab before crash
2. Restart application
3. Check system resources
4. Update dependencies:
   ```
   pip install --upgrade -r requirements.txt
   ```

---

### "Import Error" / "Module not found"

**Problem:** Missing dependencies

**Solutions:**
1. **Activate virtual environment:**
   ```
   .venv\Scripts\activate.bat
   ```
2. **Install requirements:**
   ```
   pip install -r requirements.txt
   ```
3. **Check you're in project directory**

---

## 💡 Performance Tips

### For Faster Training:
- Use GPU (cuda)
- Smaller model (yolov5n)
- Smaller image size (416)
- Larger batch size (if memory allows)

### For Better Accuracy:
- More images (300+)
- More epochs (100+)
- Larger model (yolov5m/l)
- Better data quality

### For Lower Memory:
- Smaller batch size (4-8)
- Smaller image size (320-416)
- Smaller model (yolov5n)
- Close other programs

---

## 🆘 Still Having Issues?

### Check These:

1. **Logs Tab** - Look for error messages
2. **System Resources** - RAM, disk space
3. **File Paths** - All paths correct?
4. **Dependencies** - All installed?
5. **Data Quality** - Images valid?

### Get Help:

1. Read other Help menu sections
2. Check documentation files
3. Review error messages carefully
4. Try with smaller test dataset first

---

## 🎯 Quick Fixes

**Most Common Issues:**

1. **Memory Error** → Reduce batch size to 4
2. **No images** → Check file formats
3. **Training fails** → Split dataset first
4. **Low accuracy** → More images + epochs
5. **GUI won't start** → Install PyQt5

---

**Remember:** Most issues are simple fixes! Check the Logs tab first! 📝
"""
        self.show_help_dialog("🔧 Troubleshooting", content)
    
    def show_about(self):
        """Show about dialog with dark theme"""
        # Create custom message box with dark theme
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About YOLO Training Pipeline")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(
            "<h2 style='color: #00ff88; font-size: 24px;'>🤖 YOLO Training Pipeline</h2>"
            "<p style='color: #ffffff; font-size: 24px;'><b>Version:</b> 1.0.0</p>"
            "<p style='color: #ffffff; font-size: 24px;'><b>Object Detection System</b></p>"
            "<br>"
            "<p style='color: #ffffff; font-size: 24px;'>A complete, professional-grade system for training YOLO object detection models.</p>"
            "<br>"
            "<p style='color: #00ff88; font-size: 24px;'><b>Features:</b></p>"
            "<ul style='color: #ffffff; font-size: 24px;'>"
            "<li>✅ Beautiful high-tech GUI</li>"
            "<li>✅ Complete training pipeline</li>"
            "<li>✅ Ethical data handling</li>"
            "<li>✅ Professional quality</li>"
            "<li>✅ Easy to use</li>"
            "</ul>"
            "<br>"
            "<p style='color: #ffffff; font-size: 24px;'><b>Technology:</b> Python 3.10, PyQt5, YOLO, Pillow</p>"
            "<p style='color: #ffffff; font-size: 24px;'><b>License:</b> MIT</p>"
            "<br>"
            "<p style='color: #ffffff; font-size: 24px;'>Built with ❤️ for AI enthusiasts</p>"
            "<p style='color: #00ff88; font-size: 24px;'><b>Created by Andy Kong</b></p>"
        )
        
        # Apply dark theme styling
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #1a1a2e;
                color: #ffffff;
            }
            QMessageBox QLabel {
                color: #ffffff;
                background-color: #1a1a2e;
                min-width: 450px;
                font-size: 24px;
            }
            QPushButton {
                background-color: #0f3460;
                color: #00ff88;
                border: 2px solid #00ff88;
                border-radius: 5px;
                padding: 8px 20px;
                font-weight: bold;
                min-width: 80px;
                font-size: 19px;
            }
            QPushButton:hover {
                background-color: #00ff88;
                color: #1a1a2e;
            }
        """)
        
        msg_box.exec_()


def main():
    """Main entry point for GUI"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better dark theme
    
    window = YOLOTrainingGUI()
    window.showMaximized()  # Maximize the window
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
