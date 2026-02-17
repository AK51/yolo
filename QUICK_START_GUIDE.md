# YOLO Training Pipeline - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA (optional but recommended)
- 10GB+ free disk space

### Installation

```bash
# 1. Clone and navigate to project
cd yolo-training-pipeline

# 2. Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 🎨 GUI Quick Start

### Launch GUI
```bash
python launch_gui.py
# or double-click: launch_gui_venv.bat
```

### Train Your First Model (GUI)

1. **Dataset Tab**
   - Click "Browse" to select your images folder
   - Click "Import Images"
   - Set split ratios (default: 70/20/10)
   - Click "Split Dataset"

2. **Training Tab**
   - Select YOLO version (YOLOv5 or YOLOv8)
   - Choose model size (n, s, m, l, x)
   - Set epochs (start with 50)
   - Select device (cuda for GPU)
   - Click "Start Training"

3. **Detection Tab**
   - Load your trained model (or use pre-trained)
   - Select an image
   - Click "Run Detection"
   - View results!

---

## 💻 CLI Quick Start

### Basic Workflow

```bash
# 1. Import dataset
python -m src.cli collect \
    --source local \
    --source-dir ./my_images \
    --dataset-root ./data/my_dataset

# 2. Split dataset
python -m src.cli split \
    --dataset-root ./data/my_dataset \
    --train 0.7 --val 0.2 --test 0.1

# 3. Train model
python -m src.cli train \
    --config ./configs/yolov5_default.yaml \
    --dataset ./data/my_dataset/dataset.yaml \
    --epochs 50

# 4. Evaluate model
python -m src.cli evaluate \
    --model ./model/best.pt \
    --dataset-root ./data/my_dataset
```

---

## 📝 Python API Quick Start

### Complete Example

```python
from pathlib import Path
from src.dataset import DatasetManager
from src.training import TrainingEngine
from src.models import TrainingConfig

# 1. Setup dataset
dataset_root = Path('./data/my_dataset')
manager = DatasetManager(dataset_root)

# Import images
result = manager.import_local_images(Path('./my_images'))
print(f"✅ Imported {result.images_collected} images")

# Split dataset
splits = manager.split_dataset(0.7, 0.2, 0.1)
print(f"✅ Split: Train={splits['train']}, Val={splits['val']}, Test={splits['test']}")

# 2. Train model
config = TrainingConfig(
    model_architecture='yolov5s',
    epochs=50,
    batch_size=16,
    image_size=640,
    dataset_yaml=dataset_root / 'dataset.yaml',
    device='cuda',
    output_dir='./model'
)

engine = TrainingEngine(config, config.output_dir)
result = engine.train()
print(f"✅ Training complete! mAP50: {result.final_metrics['map50']:.4f}")
print(f"📦 Model saved: {result.model_path}")
```

---

## 🎯 Common Use Cases

### Use Case 1: Train on Custom Dataset

```python
# Your images in: ./my_images/*.jpg
# Your labels in: ./my_labels/*.txt (YOLO format)

from pathlib import Path
from src.dataset import DatasetManager

# Import and organize
manager = DatasetManager('./data/custom')
manager.import_local_images('./my_images')
manager.split_dataset(0.7, 0.2, 0.1)

# Train (use GUI or CLI)
```

### Use Case 2: Convert COCO Annotations

```python
from src.annotation import AnnotationProcessor

processor = AnnotationProcessor(class_mapping={0: 'person', 1: 'car'})
result = processor.convert_coco_to_yolo(
    coco_json='./annotations.json',
    images_dir='./images',
    output_dir='./labels'
)
print(f"✅ Converted {result.annotations_converted} annotations")
```

### Use Case 3: Evaluate Existing Model

```python
from src.evaluation import EvaluationModule
from src.models import EvaluationConfig

config = EvaluationConfig(confidence_threshold=0.25)
evaluator = EvaluationModule('./model/best.pt', config)

result = evaluator.evaluate(dataset_split='test')
print(f"mAP50: {result.overall_metrics.map50:.4f}")
print(f"Precision: {result.overall_metrics.precision:.4f}")
print(f"Recall: {result.overall_metrics.recall:.4f}")
```

---

## 🔧 Configuration

### Training Configuration (YAML)

```yaml
# configs/my_training.yaml
model:
  architecture: yolov5s
  pretrained: true

training:
  epochs: 100
  batch_size: 16
  image_size: 640
  learning_rate: 0.01
  device: cuda

data:
  dataset_yaml: ./data/my_dataset/dataset.yaml
  augmentation: true

output:
  save_period: 10
  output_dir: ./model
```

### Dataset Configuration (YAML)

```yaml
# data/my_dataset/dataset.yaml
path: ./data/my_dataset
train: images/train
val: images/val
test: images/test

nc: 2  # number of classes
names: ['person', 'car']
```

---

## 📊 Monitoring Training

### GUI
- Real-time progress bar
- Live metrics display
- Training history graph

### CLI
- Watch log files: `tail -f logs/training_engine.log`
- Check training history: `cat model/training_history.json`

### Python
```python
# Access training history
import json
with open('model/training_history.json') as f:
    history = json.load(f)
    
for epoch in history['epochs']:
    print(f"Epoch {epoch['epoch']}: mAP50={epoch['map50']:.4f}")
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size
```python
config.batch_size = 8  # Instead of 16
```

### Issue: DLL Error (Windows)
**Solution**: Use correct PyTorch version
```bash
pip install torch==2.1.0+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Model Not Found
**Solution**: Check output directory
```python
# Models are saved to: output_dir/train/weights/best.pt
model_path = Path('model/train/weights/best.pt')
```

### Issue: Low mAP
**Solutions**:
1. Train longer (more epochs)
2. Use larger model (yolov5m instead of yolov5s)
3. Increase image size (640 → 1280)
4. Add more training data
5. Check annotation quality

---

## 📚 Next Steps

1. **Read Full Documentation**: `SYSTEM_DOCUMENTATION.md`
2. **Check Examples**: `example_usage.py`
3. **Review Tests**: `tests/unit/test_dataset_manager_updated.py`
4. **Explore GUI**: Try all tabs and features
5. **Experiment**: Train on your own dataset!

---

## 🎓 Learning Resources

### YOLO Resources
- [Ultralytics YOLOv5 Docs](https://docs.ultralytics.com/yolov5/)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)

### Dataset Resources
- [COCO Dataset](https://cocodataset.org/)
- [Open Images](https://storage.googleapis.com/openimages/web/index.html)
- [Roboflow Universe](https://universe.roboflow.com/)

### Annotation Tools
- [LabelImg](https://github.com/heartexlabs/labelImg)
- [CVAT](https://github.com/opencv/cvat)
- [Roboflow Annotate](https://roboflow.com/annotate)

---

## 💡 Tips & Best Practices

### Training Tips
1. **Start Small**: Use yolov5n or yolov5s for quick experiments
2. **Use Pretrained**: Always start with pretrained weights
3. **Monitor Metrics**: Watch mAP50 and loss curves
4. **Save Checkpoints**: Enable periodic saving
5. **Use GPU**: 10-50x faster than CPU

### Dataset Tips
1. **Quality > Quantity**: 100 good images > 1000 poor images
2. **Balanced Classes**: Similar number of examples per class
3. **Diverse Data**: Various lighting, angles, backgrounds
4. **Clean Labels**: Accurate bounding boxes
5. **Test Split**: Keep test set separate and untouched

### Performance Tips
1. **Batch Size**: Larger = faster (if GPU memory allows)
2. **Image Size**: 640 is good balance of speed/accuracy
3. **Workers**: Increase for faster data loading
4. **Mixed Precision**: Enable AMP for faster training
5. **Cache Dataset**: Keep on SSD for faster access

---

## ✅ Checklist

Before training:
- [ ] Dataset imported and organized
- [ ] Annotations in YOLO format
- [ ] Dataset split (train/val/test)
- [ ] Configuration file ready
- [ ] GPU available (optional)

After training:
- [ ] Model saved successfully
- [ ] Training metrics look good
- [ ] Model evaluated on test set
- [ ] Results visualized
- [ ] Model archived with timestamp

---

## 🆘 Getting Help

1. **Check Logs**: `logs/` directory
2. **Read Docs**: `SYSTEM_DOCUMENTATION.md`
3. **Review Tests**: See working examples
4. **Check Issues**: Common problems and solutions

---

**Happy Training! 🎉**

For detailed information, see `SYSTEM_DOCUMENTATION.md`
