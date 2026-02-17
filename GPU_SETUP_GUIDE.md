# GPU Training Setup Guide - RTX 4070

## Your Hardware

- **GPU:** NVIDIA GeForce RTX 4070
- **VRAM:** 8GB
- **CUDA Driver:** 12.8 (572.83)
- **Status:** ✅ GPU detected and ready

## The Problem

You have PyTorch 2.10.0+cu128 which has DLL initialization issues. We need to install a stable version.

## The Solution

### Automated Fix (Recommended)

Run this script:
```bash
fix_pytorch_gpu.bat
```

This will:
1. Uninstall PyTorch 2.10.0+cu128
2. Install PyTorch 2.1.0+cu121 (stable, compatible with CUDA 12.8)
3. Test GPU availability
4. Confirm your RTX 4070 is ready

### Manual Fix

```bash
# 1. Activate environment
.venv\Scripts\activate

# 2. Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# 3. Install PyTorch 2.1.0 with CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Reinstall Ultralytics
pip install ultralytics --upgrade

# 5. Test GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

## Verification

After running the fix, you should see:

```
PyTorch: 2.1.0+cu121
CUDA Available: True
GPU: NVIDIA GeForce RTX 4070
GPU Count: 1
```

## Using GPU in the GUI

1. Launch the GUI:
   ```bash
   python launch_gui.py
   ```

2. Go to the **Training** tab

3. In the **Device** dropdown, select **"cuda"**

4. Start training - your RTX 4070 will be used!

## Performance Expectations

With RTX 4070 (8GB VRAM):

### Recommended Settings:
- **Model:** yolov5s or yolov8n (good balance)
- **Image Size:** 640 (standard)
- **Batch Size:** 16-32 (adjust based on memory)
- **Epochs:** 50-100

### Training Speed:
- **YOLOv8n:** ~3-5 seconds per epoch (small dataset)
- **YOLOv8s:** ~5-8 seconds per epoch (small dataset)
- **YOLOv8m:** ~8-12 seconds per epoch (small dataset)

### GPU vs CPU:
- **GPU (RTX 4070):** 10-30x faster than CPU
- **Example:** 50 epochs in 5-10 minutes vs 2-3 hours on CPU

## Batch Size Guidelines

Your RTX 4070 has 8GB VRAM. Recommended batch sizes:

| Model | Image Size | Max Batch Size |
|-------|------------|----------------|
| yolov5n | 640 | 64 |
| yolov5s | 640 | 32 |
| yolov5m | 640 | 16 |
| yolov5l | 640 | 8 |
| yolov8n | 640 | 64 |
| yolov8s | 640 | 32 |

If you get "CUDA out of memory" errors, reduce batch size.

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
1. Reduce batch size (try 16, then 8, then 4)
2. Reduce image size (try 416 instead of 640)
3. Use smaller model (yolov5n instead of yolov5s)

### Issue: Still getting DLL errors

**Solution:**
1. Install Visual C++ Redistributable:
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install and restart computer

2. Try PyTorch 2.0.1 instead:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: GPU not detected

**Solution:**
```bash
# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check driver
nvidia-smi
```

## Quick Test

Test GPU training with a small dataset:

```bash
python test_real_training.py
```

This will train for 3 epochs on GPU and verify everything works.

## Monitoring GPU Usage

While training, open a new terminal and run:

```bash
nvidia-smi -l 1
```

This shows GPU usage, memory, temperature in real-time.

## Expected GPU Utilization

During training:
- **GPU Utilization:** 80-100%
- **Memory Usage:** 2-6GB (depending on batch size)
- **Temperature:** 60-80°C (normal)
- **Power:** 80-140W

## Summary

**Current Issue:** PyTorch 2.10.0+cu128 has DLL errors  
**Solution:** Install PyTorch 2.1.0+cu121 (stable)  
**Command:** `fix_pytorch_gpu.bat`  
**Result:** RTX 4070 ready for 10-30x faster training!

## After Fix

1. Run: `python launch_gui.py`
2. Training tab → Device: **cuda**
3. Start training with GPU power! 🚀
