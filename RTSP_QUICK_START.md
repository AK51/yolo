# 📡 RTSP Quick Start Guide

## 5-Minute Setup

### Step 1: Get Your RTSP URL
Find your camera's RTSP URL from its settings or manual.

**Common format:**
```
rtsp://username:password@192.168.1.100:554/stream
```

### Step 2: Test the URL
Open VLC Media Player and test the URL:
1. Open VLC
2. Media → Open Network Stream
3. Paste your RTSP URL
4. Click Play

If it works in VLC, it will work in the GUI!

### Step 3: Open RTSP Tab
1. Launch the GUI: `launch_gui_venv.bat`
2. Click on **📡 RTSP** tab

### Step 4: Configure
1. **Model Path**: Browse to your trained model (e.g., `./runs/train/best.pt`)
2. **Confidence**: Leave at 0.25 (adjust later if needed)
3. **Class Names**: Enter your classes (e.g., "baby" or "person,car,dog")
4. **RTSP URL**: Paste your camera's RTSP URL

### Step 5: Start Streaming
1. Click **▶️ Start Stream**
2. Wait 2-5 seconds for connection
3. Watch live detections! 🎉

### Step 6: Adjust Settings
- **Too many false detections?** → Increase confidence to 0.35-0.50
- **Missing detections?** → Decrease confidence to 0.15-0.20
- **Low FPS?** → Use smaller model (yolov5n) or reduce camera resolution

### Step 7: Stop When Done
Click **⏹️ Stop Stream** to disconnect.

---

## Common RTSP URLs

### Generic IP Camera
```
rtsp://192.168.1.100:554/stream
```

### With Authentication
```
rtsp://admin:password@192.168.1.100:554/stream
```

### Hikvision
```
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
```

### Dahua
```
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

### Reolink
```
rtsp://admin:password@192.168.1.100:554/h264Preview_01_main
```

---

## Troubleshooting

### Can't Connect?
1. ✅ Test URL in VLC first
2. ✅ Check camera IP (ping it)
3. ✅ Verify username/password
4. ✅ Ensure camera is on same network

### No Detections?
1. ✅ Lower confidence to 0.15
2. ✅ Check model is trained for objects in view
3. ✅ Improve lighting
4. ✅ Move camera closer

### Low FPS?
1. ✅ Use smaller model (yolov5n)
2. ✅ Reduce camera resolution
3. ✅ Close other applications
4. ✅ Use wired connection

---

## Example: Baby Monitor

**Setup:**
```
Model: ./runs/train/best.pt (trained on baby images)
Confidence: 0.30
Class Names: baby
RTSP URL: rtsp://admin:password@192.168.1.100:554/stream
```

**Result:** Real-time baby detection with bounding boxes! 👶

---

## Example: Security Camera

**Setup:**
```
Model: yolov5s.pt (pre-trained COCO)
Confidence: 0.40
Class Names: person,car,dog
RTSP URL: rtsp://admin:password@192.168.1.101:554/stream
```

**Result:** Real-time person and vehicle detection! 🚗

---

## Tips

💡 **Start with low confidence** (0.20) and increase if too many false alarms  
💡 **Test in VLC first** to verify stream works  
💡 **Use wired connection** for best stability  
💡 **Position camera well** - good lighting and angle matter  
💡 **Monitor FPS** - should be 10+ for smooth operation  

---

**Ready to go?** Open the **📡 RTSP** tab and start streaming! 🚀
