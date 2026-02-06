# 📹 USB Camera Quick Start Guide

## 3-Minute Setup

### Step 1: Connect Your Camera
Plug in your USB camera or ensure your laptop webcam is available.

### Step 2: Open USB Cam Tab
1. Launch the GUI: `launch_gui_venv.bat`
2. Click on **📹 USB Cam** tab

### Step 3: Detect Cameras
1. Click **🔍 Detect Cameras**
2. Wait for scan (2-3 seconds)
3. See available cameras listed

**Example:**
```
✅ Found: Camera 0, Camera 1
```

### Step 4: Configure
1. **Model Path**: Browse to your trained model (e.g., `./runs/train/best.pt`)
2. **Confidence**: Leave at 0.25 (adjust later if needed)
3. **Class Names**: Enter your classes (e.g., "baby" or "person,car,dog")
4. **Camera Index**: Should be auto-selected (usually 0)

### Step 5: Start Camera
1. Click **▶️ Start Camera**
2. Wait 1-2 seconds for camera to open
3. Watch live detections! 🎉

### Step 6: Adjust Settings
- **Too many false detections?** → Increase confidence to 0.35-0.50
- **Missing detections?** → Decrease confidence to 0.15-0.20
- **Low FPS?** → Use smaller model (yolov5n)

### Step 7: Stop When Done
Click **⏹️ Stop Camera** to close the camera.

---

## Camera Index Quick Reference

- **0** = Built-in laptop webcam (most common)
- **1** = First external USB camera
- **2** = Second external USB camera
- **3+** = Additional cameras

**Tip:** Use **🔍 Detect Cameras** to find all available cameras!

---

## Troubleshooting

### No Cameras Found?
1. ✅ Check camera is plugged in
2. ✅ Close Zoom, Skype, Teams, etc.
3. ✅ Try different USB port
4. ✅ Restart computer

### Camera Won't Open?
1. ✅ Try different camera index (0, 1, 2...)
2. ✅ Close other camera apps
3. ✅ Run as administrator
4. ✅ Update camera drivers

### No Detections?
1. ✅ Lower confidence to 0.15
2. ✅ Improve lighting
3. ✅ Move camera closer
4. ✅ Check model is trained for objects in view

### Low FPS?
1. ✅ Use smaller model (yolov5n)
2. ✅ Close other applications
3. ✅ Ensure good lighting

---

## Example: Baby Monitor

**Setup:**
```
Model: ./runs/train/best.pt (trained on baby images)
Confidence: 0.30
Class Names: baby
Camera Index: 0 (built-in webcam)
```

**Steps:**
1. Position laptop to see crib
2. Detect cameras → Select Camera 0
3. Load baby detection model
4. Start camera
5. Monitor in real-time! 👶

---

## Example: Desk Occupancy

**Setup:**
```
Model: yolov5s.pt (pre-trained COCO)
Confidence: 0.40
Class Names: person
Camera Index: 1 (external USB camera)
```

**Steps:**
1. Position USB camera over desk
2. Detect cameras → Select Camera 1
3. Load COCO model
4. Start camera
5. Track desk usage! 💼

---

## Tips

💡 **Always detect cameras first** to see what's available  
💡 **Start with confidence 0.25** and adjust based on results  
💡 **Close other camera apps** (Zoom, Skype) before starting  
💡 **Use good lighting** for best detection accuracy  
💡 **Position camera well** - stable mount, good angle  

---

## Comparison with RTSP

### Use USB Cam When:
- Testing locally
- Using laptop webcam
- No network setup needed
- Quick demonstrations
- Personal projects

### Use RTSP When:
- Remote monitoring
- Professional cameras
- Multiple locations
- Production deployments
- High-quality requirements

---

**Ready to go?** Open the **📹 USB Cam** tab and click 'Detect Cameras'! 🚀
