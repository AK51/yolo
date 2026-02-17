# 🎨 YOLO Training Pipeline - GUI Guide

## High-Tech GUI Interface

A modern, sleek PyQt5 GUI for the YOLO Training Pipeline with a cyberpunk-inspired dark theme.

## 🚀 Quick Start

### Launch the GUI

**Option 1: Double-click the batch file**
```
launch_gui.bat
```

**Option 2: Command line**
```cmd
.venv\Scripts\activate.bat
python launch_gui.py
```

**Option 3: Direct Python**
```cmd
.venv\Scripts\python.exe launch_gui.py
```

## 🎯 GUI Features

### 📁 Dataset Tab

**Data Collection:**
1. Set **Dataset Root** (where your dataset will be stored)
   - Default: `./data/baby_dataset`
   - Click "📂 Browse" to select a folder

2. Set **Source Directory** (where your images are)
   - Click "📂 Browse" to select your image folder
   - Supports: .jpg, .jpeg, .png, .bmp

3. Click **🚀 Collect Images**
   - Validates all images
   - Removes duplicates
   - Organizes into dataset structure

**Dataset Split:**
1. Set split ratios (must sum to 1.0):
   - **Train Ratio**: 0.7 (70%)
   - **Val Ratio**: 0.2 (20%)
   - **Test Ratio**: 0.1 (10%)

2. Click **✂️ Split Dataset**
   - Randomly splits images
   - Moves to train/val/test folders

**Statistics:**
- View total images, size, and split distribution
- Click **🔄 Refresh Statistics** to update

### 🚀 Training Tab

**Model Configuration:**
1. **YOLO Version**: Choose yolov5 or yolov8
2. **Model Architecture**: 
   - yolov5n (nano - fastest)
   - yolov5s (small - balanced)
   - yolov5m (medium)
   - yolov5l (large)
   - yolov5x (extra large - most accurate)

3. **Training Parameters:**
   - **Epochs**: Number of training iterations (50-100 recommended)
   - **Batch Size**: Images per batch (16 default, reduce if memory issues)
   - **Image Size**: Input image size (640 recommended)
   - **Device**: cpu or cuda (GPU)
   - **Output Directory**: Where to save trained model

4. Click **🚀 Start Training**
   - Training runs in background
   - Progress shown in logs
   - Model saved when complete

### 📊 Evaluation Tab

**Model Evaluation:**
1. **Model Path**: Path to trained model (.pt file)
   - Default: `./runs/train/best.pt`
   - Click "📂 Browse" to select

2. **Dataset Split**: Choose test or val set

3. **Confidence Threshold**: Detection confidence (0.25 default)

4. Click **📊 Evaluate Model**
   - Runs inference on selected dataset
   - Shows metrics: Precision, Recall, mAP50, mAP50-95
   - Displays inference time

### 📝 Logs Tab

**System Logs:**
- Real-time logging of all operations
- Timestamps for each action
- Success ✅ and error ❌ indicators
- Click **🗑️ Clear Logs** to reset

## 🎨 GUI Theme

**High-Tech Dark Theme:**
- Cyberpunk-inspired color scheme
- Neon green (#00ff88) accents
- Dark blue (#1a1a2e) background
- Smooth animations and hover effects
- Professional, modern look

**Color Palette:**
- Background: Dark Navy (#1a1a2e)
- Primary: Neon Green (#00ff88)
- Secondary: Deep Blue (#16213e)
- Accent: Cyan (#00ffff)
- Text: White (#ffffff)

## 📖 Step-by-Step Workflow

### Complete Training Workflow

1. **Launch GUI**
   ```
   launch_gui.bat
   ```

2. **Collect Images** (Dataset Tab)
   - Browse to your image folder
   - Click "🚀 Collect Images"
   - Wait for completion message

3. **Check Statistics** (Dataset Tab)
   - Click "🔄 Refresh Statistics"
   - Verify image count

4. **Split Dataset** (Dataset Tab)
   - Adjust ratios if needed
   - Click "✂️ Split Dataset"
   - Verify split in statistics

5. **Configure Training** (Training Tab)
   - Select YOLO version
   - Choose model architecture
   - Set epochs (start with 10 for testing)
   - Set batch size (8 or 16)
   - Keep device as "cpu" unless you have GPU

6. **Start Training** (Training Tab)
   - Click "🚀 Start Training"
   - Monitor progress in Logs tab
   - Wait for completion (can take time)

7. **Evaluate Model** (Evaluation Tab)
   - Model path auto-filled
   - Select "test" split
   - Click "📊 Evaluate Model"
   - Review metrics

## 💡 Tips & Tricks

### Performance Tips

**For Faster Training:**
- Use smaller model (yolov5n)
- Reduce epochs (10-20 for testing)
- Reduce batch size (4-8)
- Use smaller image size (416)

**For Better Accuracy:**
- Use larger model (yolov5m or yolov5l)
- Increase epochs (100-300)
- Use larger image size (640-1280)
- Collect more images (300+)

### Memory Management

**If you get memory errors:**
1. Reduce batch size: 16 → 8 → 4 → 2
2. Reduce image size: 640 → 416 → 320
3. Use smaller model: yolov5s → yolov5n
4. Close other applications

### Dataset Quality

**For best results:**
- Use 300+ images
- Include variety (angles, lighting, backgrounds)
- Ensure good image quality
- Remove blurry or corrupted images
- Balance your dataset

## 🔧 Troubleshooting

### GUI Won't Start

**Error: "No module named PyQt5"**
```cmd
.venv\Scripts\python.exe -m pip install PyQt5
```

**Error: "No module named src"**
- Make sure you're in the project directory
- Run from the project root directory

### Collection Issues

**"Source directory does not exist"**
- Check the path is correct
- Use "📂 Browse" button
- Ensure folder contains images

**"No images collected"**
- Check image formats (.jpg, .png, .bmp)
- Verify images meet minimum size (32x32)
- Check file permissions

### Training Issues

**"Training failed"**
- Check dataset is split
- Verify images exist in train folder
- Reduce batch size if memory error
- Check logs for specific error

### Evaluation Issues

**"Model file does not exist"**
- Check model path is correct
- Ensure training completed successfully
- Look in output directory for .pt file

## 🎮 Keyboard Shortcuts

- **Ctrl+Q**: Quit application
- **F5**: Refresh statistics (when in Dataset tab)
- **Ctrl+L**: Clear logs (when in Logs tab)

## 📊 Understanding Metrics

**Precision**: How many detections were correct
- Higher is better (0.0 - 1.0)
- 0.85 = 85% of detections were correct

**Recall**: How many objects were detected
- Higher is better (0.0 - 1.0)
- 0.80 = 80% of objects were found

**mAP50**: Mean Average Precision at 50% IoU
- Primary metric for object detection
- Higher is better (0.0 - 1.0)
- 0.75 = Good performance

**F1 Score**: Balance of precision and recall
- Higher is better (0.0 - 1.0)
- Harmonic mean of precision and recall

## 🌟 Advanced Features

### Custom Configuration

Edit configuration files before training:
- `configs/yolov5_default.yaml`
- `configs/yolov8_default.yaml`

### Batch Processing

Process multiple datasets:
1. Collect first dataset
2. Train model
3. Change dataset root
4. Collect second dataset
5. Train new model

### Model Comparison

Compare different models:
1. Train with yolov5n → Evaluate
2. Train with yolov5s → Evaluate
3. Train with yolov5m → Evaluate
4. Compare metrics

## 🎯 Best Practices

1. **Always start with a small test**
   - 10 epochs
   - Small batch size
   - Verify everything works

2. **Monitor the logs**
   - Watch for errors
   - Check progress messages
   - Verify completion

3. **Save your work**
   - Note successful configurations
   - Keep track of best models
   - Document your results

4. **Iterate and improve**
   - Start simple
   - Gradually increase complexity
   - Learn from each training run

## 🚀 Ready to Go!

Your high-tech GUI is ready! Just double-click `launch_gui.bat` and start training your YOLO models with style! 🎨✨

---

**Need help?** Check the main README.md or QUICKSTART.md for more information.
