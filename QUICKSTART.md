# Quick Start Guide

This guide will help you get started with the YOLO Training Pipeline quickly.

## Prerequisites

- Python 3.10 installed at: `C:\Users\andyk\AppData\Local\Programs\Python\Python310\`
- Virtual environment created at: `E:\test\Kiro_baby\.venv`

## Step 1: Activate Virtual Environment

```cmd
E:\test\Kiro_baby\.venv\Scripts\activate.bat
```

Or in PowerShell:
```powershell
E:\test\Kiro_baby\.venv\Scripts\Activate.ps1
```

## Step 2: Verify Installation

Check that the CLI works:
```bash
python -m src.cli --help
```

You should see available commands: collect, split, annotate, train, evaluate, stats

## Step 3: Prepare Your Dataset

### Option A: Use Local Images

If you have images on your computer:

```bash
python -m src.cli collect --dataset-root ./data/baby_dataset --source local --source-dir C:\path\to\your\images
```

### Option B: Use Public Datasets (COCO, Open Images)

Note: COCO and Open Images collection require additional implementation. For now, use local images.

## Step 4: Check Dataset Statistics

```bash
python -m src.cli stats --dataset-root ./data/baby_dataset
```

This shows:
- Total number of images
- Dataset size
- Train/val/test split counts

## Step 5: Split Dataset

Split your dataset into training, validation, and test sets:

```bash
python -m src.cli split --dataset-root ./data/baby_dataset --train 0.7 --val 0.2 --test 0.1 --seed 42
```

This creates:
- 70% training images
- 20% validation images
- 10% test images

## Step 6: Prepare Annotations (if needed)

If you have annotations in COCO or Pascal VOC format, convert them to YOLO format:

### From COCO:
```bash
python -m src.cli annotate --format coco --input annotations.json --images-dir ./data/baby_dataset/images/train --output ./data/baby_dataset/labels/train
```

### From Pascal VOC:
```bash
python -m src.cli annotate --format voc --input ./voc_annotations --images-dir ./data/baby_dataset/images/train --output ./data/baby_dataset/labels/train
```

## Step 7: Train Your Model

### Using Default Configuration:

```bash
python -m src.cli train --yolo-version yolov5 --output ./runs/train --epochs 50 --batch-size 16 --device cpu
```

### Using Custom Configuration:

1. Edit `configs/yolov5_default.yaml` or `configs/yolov8_default.yaml`
2. Run:
```bash
python -m src.cli train --config configs/yolov5_default.yaml --output ./runs/train
```

Training parameters:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Image size (default: 640)
- `--device`: Device to use (cpu, cuda)

## Step 8: Evaluate Your Model

After training completes:

```bash
python -m src.cli evaluate --model ./runs/train/best.pt --split test --report ./evaluation_report.md
```

This will:
- Run inference on test set
- Calculate metrics (precision, recall, mAP)
- Generate evaluation report

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Activate environment
E:\test\Kiro_baby\.venv\Scripts\activate.bat

# 2. Collect images
python -m src.cli collect --dataset-root ./data/baby_dataset --source local --source-dir C:\Users\YourName\Pictures\babies

# 3. Check stats
python -m src.cli stats --dataset-root ./data/baby_dataset

# 4. Split dataset
python -m src.cli split --dataset-root ./data/baby_dataset --train 0.7 --val 0.2 --test 0.1

# 5. Train model (quick test with 10 epochs)
python -m src.cli train --yolo-version yolov5 --output ./runs/train --epochs 10 --batch-size 8 --device cpu

# 6. Evaluate
python -m src.cli evaluate --model ./runs/train/best.pt --split test --report ./report.md
```

## Important Notes

### About Training

This pipeline provides the structure for YOLO training. For actual YOLO model training, you need to:

1. Install YOLOv5:
```bash
pip install yolov5
```

2. Or install YOLOv8:
```bash
pip install ultralytics
```

The current implementation provides a simulation of training. For production use, integrate with actual YOLO libraries.

### About Data Collection

**IMPORTANT**: When collecting images of babies or children:
- Ensure you have proper rights and permissions
- Respect privacy laws and regulations
- Use only images you have legal rights to use
- Consider using public datasets with proper licensing
- Never scrape images without permission

### Recommended Datasets

For object detection training, consider these ethical sources:
- **COCO Dataset**: http://cocodataset.org/
- **Open Images**: https://storage.googleapis.com/openimages/web/index.html
- **Your own images**: With proper consent and rights

## Troubleshooting

### Import Errors
If you get import errors, make sure you're in the project root directory and the virtual environment is activated.

### Memory Issues
If training fails due to memory:
- Reduce `--batch-size` (try 8, 4, or 2)
- Reduce `--img-size` (try 416 or 320)
- Use CPU instead of GPU if GPU memory is insufficient

### No Images Found
Make sure your image directory contains supported formats:
- .jpg, .jpeg
- .png
- .bmp

## Next Steps

1. Collect more images to improve model accuracy (aim for 300+ images)
2. Add proper annotations for your images
3. Experiment with different YOLO versions (yolov5s, yolov8n, etc.)
4. Adjust training parameters (epochs, batch size, learning rate)
5. Evaluate and iterate on your model

## Getting Help

- Check the main README.md for detailed documentation
- Run any command with `--help` to see options
- Review example_usage.py for code examples

Happy training! 🚀
