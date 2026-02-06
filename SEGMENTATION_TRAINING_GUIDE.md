# Segmentation Training Quick Start Guide

## What's New? 🎉

The Training tab now supports **Instance Segmentation** in addition to Object Detection!

## Visual Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ⚙️ Model Configuration                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Task Type:          [Object Detection ▼]  YOLO Version: [yolov5 ▼]  │
│                                                              │
│  Model Architecture: [yolov5s ▼]           Device:       [cuda ▼]     │
│                                                              │
│  Image Size:         [640 ▼]                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step: Train a Segmentation Model

### Step 1: Select Task Type
Click the **Task Type** dropdown and select **"Segmentation"**

```
Task Type: [Segmentation ▼]
```

### Step 2: Choose YOLO Version
Select your preferred YOLO version:
- yolov5 (faster, smaller)
- yolov8 (balanced)
- yolo11 (latest, most accurate)

```
YOLO Version: [yolo11 ▼]
```

### Step 3: Model Architecture Updates Automatically
When you select "Segmentation", the model list changes to show segmentation models:

**Before (Object Detection):**
```
Model Architecture: [yolo11n, yolo11s, yolo11m, yolo11l, yolo11x]
```

**After (Segmentation):**
```
Model Architecture: [yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg]
```

### Step 4: Configure Training Parameters
Set your training parameters as usual:
- Epochs: 50
- Batch Size: 16
- Learning Rate: 0.01
- etc.

### Step 5: Prepare Your Dataset
Make sure your dataset has polygon annotations:

**Dataset Structure:**
```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       └── image3.jpg
└── labels/
    ├── train/
    │   ├── image1.txt  (polygon format)
    │   └── image2.txt  (polygon format)
    └── val/
        └── image3.txt  (polygon format)
```

**Label Format (Segmentation):**
```
0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4
```
Format: `class_id x1 y1 x2 y2 x3 y3 ...`

### Step 6: Start Training
Click **"🚀 Start Training"**

The training log will show:
```
==================================================
TRAINING CONFIGURATION
==================================================
Task Type: Segmentation
YOLO Version: yolo11
Model Architecture: yolo11l-seg
Device: cuda
Image Size: 640
Epochs: 50
Batch Size: 16
...
==================================================
```

## Model Comparison

### Object Detection Models
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolo11n | Nano | ⚡⚡⚡ | ⭐⭐ | Mobile, Edge devices |
| yolo11s | Small | ⚡⚡ | ⭐⭐⭐ | General purpose |
| yolo11m | Medium | ⚡ | ⭐⭐⭐⭐ | Balanced |
| yolo11l | Large | 🐌 | ⭐⭐⭐⭐⭐ | High accuracy |
| yolo11x | XLarge | 🐌🐌 | ⭐⭐⭐⭐⭐⭐ | Maximum accuracy |

### Segmentation Models (add -seg suffix)
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolo11n-seg | Nano | ⚡⚡⚡ | ⭐⭐ | Mobile segmentation |
| yolo11s-seg | Small | ⚡⚡ | ⭐⭐⭐ | General segmentation |
| yolo11m-seg | Medium | ⚡ | ⭐⭐⭐⭐ | Balanced segmentation |
| yolo11l-seg | Large | 🐌 | ⭐⭐⭐⭐⭐ | High accuracy segmentation |
| yolo11x-seg | XLarge | 🐌🐌 | ⭐⭐⭐⭐⭐⭐ | Maximum accuracy segmentation |

## Creating Segmentation Datasets

### Using the Labeling Tab

1. **Open Labeling Tab**
2. **Load Images**: Click "Browse" and select your images folder
3. **Select Drawing Mode**: Choose **"Polygon"** (not Bounding Box)
4. **Draw Polygons**:
   - Click to add points around the object
   - Right-click or click near first point to close polygon
   - Polygon will show with semi-transparent fill
5. **Select Class**: Choose the class from the list
6. **Save**: Click "💾 Save" or press Ctrl+S
7. **Next Image**: Click "Next" to label more images

### Polygon Drawing Tips
- Click carefully around object edges
- More points = more accurate segmentation
- Use Ctrl+Z to undo last point
- Use "Clear All" to start over
- Polygons are saved in YOLO segmentation format automatically

## Training Tips

### For Segmentation Models:
1. **More Data**: Segmentation needs more training data than detection
2. **Higher Epochs**: Consider 100-200 epochs for better results
3. **GPU Recommended**: Segmentation is more computationally intensive
4. **Batch Size**: Reduce if you get out-of-memory errors
5. **Image Size**: 640 is good, 1280 for higher accuracy (slower)

### Recommended Settings:

**Fast Training (Testing):**
```
Task Type: Segmentation
Model: yolo11n-seg
Epochs: 50
Batch Size: 16
Image Size: 640
```

**Production Quality:**
```
Task Type: Segmentation
Model: yolo11l-seg
Epochs: 150
Batch Size: 8
Image Size: 640
```

**Maximum Accuracy:**
```
Task Type: Segmentation
Model: yolo11x-seg
Epochs: 200
Batch Size: 4
Image Size: 1280
```

## Testing Your Segmentation Model

### Test Tab
1. Go to **Test** tab
2. Load your trained segmentation model (e.g., `model/train/weights/best.pt`)
3. Load test image
4. Click "Run Detection"
5. See segmentation masks overlaid on image

### USB Cam Tab
1. Go to **USB Cam** tab
2. Select **Detection Mode**: "Segmentation"
3. Load your segmentation model
4. Click "Start Camera"
5. See real-time segmentation with colored masks

### RTSP Tab
1. Go to **RTSP** tab
2. Load your segmentation model
3. Enter RTSP URL
4. Click "Connect"
5. See real-time segmentation on video stream

## Troubleshooting

### Issue: "Model not found"
**Solution**: The model will be downloaded automatically on first use. Ensure internet connection.

### Issue: "Out of memory"
**Solution**: Reduce batch size (try 8, 4, or 2)

### Issue: "Training too slow"
**Solution**: 
- Use smaller model (yolo11n-seg instead of yolo11l-seg)
- Reduce image size (640 instead of 1280)
- Use GPU instead of CPU

### Issue: "Poor segmentation quality"
**Solution**:
- Increase training epochs
- Use larger model
- Add more training data
- Improve polygon annotations quality

## What You'll Get

After training completes, you'll have:

```
model/
└── train/
    ├── weights/
    │   ├── best.pt          ← Your trained segmentation model
    │   └── last.pt
    ├── results.png          ← Training curves
    ├── confusion_matrix.png
    ├── val_batch0_pred.jpg  ← Segmentation predictions
    └── ...
```

## Next Steps

1. ✅ Train your segmentation model
2. ✅ Test on images (Test tab)
3. ✅ Try real-time segmentation (USB Cam tab)
4. ✅ Deploy on RTSP streams (RTSP tab)
5. ✅ Evaluate performance (Evaluation tab)

## Summary

- **Task Type dropdown**: Switch between Detection and Segmentation
- **Automatic model updates**: Models change based on task type
- **Same workflow**: Training process is identical
- **Full integration**: Works with all tabs (Test, USB Cam, RTSP)
- **Easy to use**: Just select "Segmentation" and train!

Happy segmenting! 🎨🤖
