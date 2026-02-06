# YOLO Object Detection Training Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)

A complete, professional-grade system for training custom YOLO object detection models with an intuitive GUI, ethical data sourcing, and comprehensive dataset management.

![YOLO Training Pipeline](https://img.shields.io/badge/YOLO-v5%20%7C%20v8%20%7C%20v11-orange)

## ✨ Features

- 🎨 **Modern GUI** - High-tech dark theme with intuitive interface
- 🏷️ **Interactive Labeling** - Visual bounding box annotation tool (YOLO format)
- 📁 **Dataset Management** - Collection, validation, splitting, and organization
- 🚀 **Training Pipeline** - Support for YOLOv5, YOLOv8, and YOLOv11
- 🎯 **Object Detection & Segmentation** - Both detection and segmentation tasks
- 📊 **Real-time Monitoring** - Live training progress and metrics
- 🧪 **Testing Tools** - Image, video, USB camera, and RTSP stream detection
- 📈 **Evaluation Module** - Comprehensive metrics and visualizations
- 🔧 **Flexible Configuration** - Easy customization for different use cases
- 💻 **CLI & GUI** - Choose command-line or graphical interface

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/yolo-training-pipeline.git
cd yolo-training-pipeline
```

2. **Create virtual environment**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Launch GUI

**Windows:**
```cmd
launch_gui.bat
```

**Linux/Mac:**
```bash
python launch_gui.py
```

### First Training

1. **Launch the application** (see above)
2. **Prepare your dataset** - Use the Dataset tab to collect and organize images
3. **Label your images** - Use the Labeling tab to annotate objects
4. **Split your dataset** - Use the Dataset tab to split into train/val/test
5. **Go to Training tab** - Configure your training parameters
6. **Click "Start Training"** - Watch your model train in real-time!

See [START_HERE.md](START_HERE.md) for detailed quick start guide.

## 📚 Documentation

- **[Quick Start Guide](QUICK_START_GUIDE.md)** - Get started in 5 minutes
- **[GUI Guide](GUI_GUIDE.md)** - Complete GUI documentation
- **[Labeling Guide](LABELING_GUIDE.md)** - How to annotate images
- **[System Documentation](SYSTEM_DOCUMENTATION.md)** - Technical reference
- **[Dataset Structure](YOLO_DATASET_STRUCTURE.md)** - YOLO format explained

## 🎨 GUI Overview

The application features 7 main tabs:

### 1. 📁 Dataset Tab
- Collect and organize images
- Split into train/val/test sets
- View dataset statistics
- Generate YAML configuration files

### 2. 🏷️ Labeling Tab
- Interactive bounding box annotation
- YOLO format labels
- Class management
- Keyboard shortcuts for efficiency

### 3. 🚀 Training Tab
- Configure training parameters
- Support for detection and segmentation
- Real-time progress monitoring
- Automatic model archiving

### 4. 📊 Evaluation Tab
- Calculate precision, recall, mAP
- Generate performance reports
- Visualize predictions

### 5. 🧪 Test Tab
- Test on images and videos
- Real-time detection display
- Adjustable confidence thresholds

### 6. 📹 USB Camera Tab
- Live detection from webcam
- Real-time performance
- Recording capabilities

### 7. 🌐 RTSP Stream Tab
- Network camera support
- Remote monitoring
- Stream recording

## 🛠️ CLI Usage

### Collect Dataset
```bash
python -m src.cli collect --dataset-root ./data/my_dataset --source local --source-dir /path/to/images
```

### Split Dataset
```bash
python -m src.cli split --dataset-root ./data/my_dataset --train 0.7 --val 0.2 --test 0.1
```

### Train Model
```bash
python -m src.cli train --config configs/yolov8_default.yaml --output ./runs/train
```

### Evaluate Model
```bash
python -m src.cli evaluate --model ./runs/train/best.pt --split test
```

## 📂 Project Structure

```
yolo-training-pipeline/
├── src/                      # Source code
│   ├── gui/                 # PyQt5 GUI components
│   │   ├── main_window.py  # Main application window
│   │   └── image_canvas.py # Interactive labeling canvas
│   ├── dataset/            # Dataset management
│   ├── annotation/         # Annotation processing
│   ├── training/           # Training engine
│   ├── evaluation/         # Evaluation module
│   └── cli.py             # Command-line interface
├── configs/               # Configuration files
├── data/                 # Dataset storage
├── model/               # Trained models
├── tests/              # Test suite
├── docs/              # Additional documentation
├── launch_gui.py     # GUI launcher
├── launch_gui.bat   # Windows launcher
└── requirements.txt # Python dependencies
```

## 🔧 Configuration

Example training configuration (`configs/yolov8_default.yaml`):

```yaml
model_architecture: yolov8n
num_classes: 1
class_names:
  - object

epochs: 100
batch_size: 16
image_size: 640
learning_rate: 0.01
device: cuda  # or 'cpu'
```

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📋 Requirements

- Python 3.8+
- PyQt5
- PyTorch
- Ultralytics YOLO
- OpenCV
- Pillow
- NumPy

See `requirements.txt` for complete list.

## ⚠️ Important Notes

### Ethical Data Collection

This pipeline emphasizes ethical data sourcing:
- ✅ Use publicly available datasets with proper licensing (COCO, Open Images)
- ✅ Ensure you have rights to use any images you collect
- ✅ Respect privacy and copyright laws
- ✅ For images of people, ensure proper consent and legal compliance

### GPU Support

For faster training, use a CUDA-compatible GPU:
- Install CUDA toolkit
- Install PyTorch with CUDA support
- Set `device: cuda` in configuration

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO implementation
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [COCO Dataset](https://cocodataset.org/) - Sample dataset

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for the Computer Vision community**
