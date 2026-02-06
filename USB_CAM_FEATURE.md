# 📹 USB Camera Real-Time Detection Feature

## Overview

The USB Camera Real-Time Detection feature allows you to use your computer's built-in webcam or external USB cameras for live object detection. Perfect for testing, demonstrations, and local monitoring applications.

## Features

✅ **Real-Time Detection**: Process live video from USB cameras with YOLO  
✅ **Multiple Camera Support**: Detect and switch between multiple cameras  
✅ **Auto-Detection**: Automatically scan for available cameras  
✅ **Live Visualization**: See bounding boxes and labels in real-time  
✅ **Performance Metrics**: Monitor FPS and detection count  
✅ **Configurable Settings**: Adjust confidence and camera selection  
✅ **Easy Setup**: No network configuration required  

## How to Use

### 1. Prepare Your Model

Before using USB camera detection, you need a trained YOLO model:

1. Train a model using the **🚀 Training** tab
2. Or use a pre-trained model (e.g., `./runs/train/best.pt`)
3. Note the model path for later

### 2. Connect Your Camera

Ensure your camera is:
- Physically connected to your computer (USB port)
- Recognized by your operating system
- Not in use by another application (Zoom, Skype, etc.)

**Common camera types:**
- Built-in laptop webcam (usually index 0)
- External USB webcam (usually index 1+)
- USB capture cards
- USB microscopes

### 3. Detect Available Cameras

In the **📹 USB Cam** tab:

1. Click **🔍 Detect Cameras**
2. Wait for scan to complete (2-3 seconds)
3. See list of available cameras
4. Camera index will be auto-selected

**Example output:**
```
✅ Found: Camera 0, Camera 1
```

### 4. Configure Detection Settings

1. **Model Path**: Browse and select your trained model
2. **Confidence**: Set detection threshold (0.25 default)
   - Lower (0.15): More detections, more false positives
   - Higher (0.5): Fewer detections, higher accuracy
3. **Class Names**: Enter comma-separated class names (e.g., "baby,person,dog")
4. **Camera Index**: Select camera (0 = default, 1 = second camera, etc.)

### 5. Start Camera

1. Click **▶️ Start Camera**
2. Wait for camera to open (1-2 seconds)
3. Watch live detections appear on screen!

### 6. Monitor Performance

The interface shows:
- **🔴 LIVE**: Camera status indicator
- **Detections**: Number of objects detected in current frame
- **FPS**: Frames processed per second

### 7. Stop Camera

Click **⏹️ Stop Camera** when finished.

## Camera Index Guide

### Understanding Camera Indices

Cameras are numbered starting from 0:

- **Index 0**: Usually the built-in webcam (laptops)
- **Index 1**: First external USB camera
- **Index 2**: Second external USB camera
- **Index 3+**: Additional cameras

### Finding Your Camera

**Method 1: Auto-Detection (Recommended)**
1. Click **🔍 Detect Cameras**
2. See which cameras are found
3. Try each one to see which is which

**Method 2: Trial and Error**
1. Start with index 0
2. If wrong camera, try index 1
3. Continue until you find the right one

**Method 3: Device Manager (Windows)**
1. Open Device Manager
2. Expand "Cameras" or "Imaging devices"
3. See list of connected cameras
4. First in list = index 0

## Performance Tips

### For Better FPS:
- Use GPU (CUDA) if available
- Use smaller model (yolov5n)
- Close other camera applications
- Reduce camera resolution
- Ensure good lighting

### For Better Accuracy:
- Use larger model (yolov5s or yolov5m)
- Increase confidence threshold
- Ensure good lighting
- Position camera properly
- Train model with similar scenes

### For Lower CPU Usage:
- Use smaller model (yolov5n)
- Lower camera resolution
- Close other applications
- Use hardware acceleration

## Troubleshooting

### "No cameras found"

**Possible causes:**
1. Camera not connected
2. Camera drivers not installed
3. Camera in use by another app
4. USB port not working

**Solutions:**
- Check physical connection
- Install camera drivers
- Close Zoom, Skype, Teams, etc.
- Try different USB port
- Restart computer

### "Failed to open USB camera"

**Possible causes:**
1. Camera already in use
2. Wrong camera index
3. Permission issues
4. Driver problems

**Solutions:**
- Close other camera apps
- Try different camera index
- Run as administrator
- Update camera drivers
- Restart camera (unplug/replug)

### "Failed to read frame from camera"

**Possible causes:**
1. Camera disconnected
2. USB cable loose
3. Power issues
4. Camera malfunction

**Solutions:**
- Check USB connection
- Try different USB cable
- Use powered USB hub
- Test camera in other apps
- Restart camera

### Low FPS / Laggy Video

**Possible causes:**
1. CPU/GPU overload
2. High resolution camera
3. Large model (yolov5l/x)
4. Other apps using resources

**Solutions:**
- Use smaller model (yolov5n)
- Close other applications
- Lower camera resolution
- Use GPU acceleration
- Reduce confidence threshold

### No Detections Appearing

**Possible causes:**
1. Confidence threshold too high
2. Model not trained for objects in view
3. Poor lighting conditions
4. Objects too small/far

**Solutions:**
- Lower confidence to 0.15-0.20
- Retrain model with similar scenes
- Improve lighting
- Move camera closer to objects
- Use better quality camera

### Camera Shows Black Screen

**Possible causes:**
1. Camera lens covered
2. Camera privacy shutter closed
3. Camera not initialized
4. Driver issues

**Solutions:**
- Check lens is uncovered
- Open privacy shutter
- Restart application
- Update drivers
- Test in other apps

## Use Cases

### 🏠 Home Applications
- Baby monitoring
- Pet monitoring
- Home security
- Package delivery detection
- Visitor detection

### 💼 Office Applications
- Meeting room occupancy
- Desk availability
- Safety compliance
- Access control
- Visitor logging

### 🎓 Educational
- AI/ML demonstrations
- Computer vision projects
- Student presentations
- Research experiments
- Proof of concepts

### 🔬 Research
- Behavior analysis
- Object tracking
- Gesture recognition
- Activity recognition
- Data collection

## Technical Details

### Camera Settings
- **Default Resolution**: 640x480 (VGA)
- **Default FPS**: 30 FPS
- **Format**: RGB24
- **Auto-adjustment**: Brightness, contrast, exposure

### Supported Cameras
- UVC (USB Video Class) cameras
- Built-in laptop webcams
- External USB webcams
- USB capture cards
- USB microscopes
- USB endoscopes

### Performance
- **Typical FPS**: 15-30 FPS (depends on model and hardware)
- **Latency**: 30-100ms
- **CPU Usage**: 20-50% (depends on model)
- **Memory**: 100-300 MB

### Compatibility
- **Windows**: Full support
- **Linux**: Full support (requires v4l2)
- **macOS**: Full support

## Best Practices

### Camera Positioning
1. **Height**: Eye level or slightly above
2. **Angle**: Straight on or slight downward angle
3. **Distance**: 1-3 meters from subjects
4. **Stability**: Use tripod or stable mount

### Lighting
1. **Brightness**: Well-lit environment
2. **Direction**: Front lighting (avoid backlighting)
3. **Consistency**: Avoid flickering lights
4. **Natural**: Use natural light when possible

### Performance
1. **Start with low settings** and increase gradually
2. **Monitor FPS** - should be 10+ for smooth operation
3. **Close unnecessary apps** to free resources
4. **Use GPU** if available for better performance

### Privacy
1. **Cover camera** when not in use
2. **Close application** when done
3. **Be aware** of what's in frame
4. **Respect privacy** of others

## Advanced Configuration

### Custom Resolution
To change camera resolution, modify USBCamThread:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### Custom FPS
To change camera FPS:
```python
cap.set(cv2.CAP_PROP_FPS, 60)
```

### Camera Properties
Available properties:
- `CAP_PROP_BRIGHTNESS`
- `CAP_PROP_CONTRAST`
- `CAP_PROP_SATURATION`
- `CAP_PROP_HUE`
- `CAP_PROP_EXPOSURE`
- `CAP_PROP_AUTOFOCUS`

## Comparison: USB Cam vs RTSP

### USB Camera Advantages:
✅ No network configuration required  
✅ Lower latency (30-100ms)  
✅ Easier setup  
✅ No bandwidth concerns  
✅ More reliable connection  

### RTSP Advantages:
✅ Remote access  
✅ Multiple viewers  
✅ Professional cameras  
✅ Better image quality  
✅ PTZ control  

### When to Use USB Cam:
- Local monitoring
- Testing and development
- Demonstrations
- Personal projects
- Quick prototypes

### When to Use RTSP:
- Remote monitoring
- Professional installations
- Multiple locations
- High-quality requirements
- Production deployments

## Limitations

- **Single Camera**: One camera at a time
- **No Recording**: Live view only (no built-in recording)
- **No Audio**: Video only, no audio processing
- **USB Dependent**: Requires physical connection
- **Limited Range**: USB cable length (5m max)

## Future Enhancements

Potential improvements:
- Multi-camera support (split screen)
- Recording functionality
- Snapshot capture
- Camera settings adjustment
- Motion detection
- Time-lapse mode
- Video filters

## Example Workflows

### Baby Monitor Setup

1. **Position Camera**: Point at crib/play area
2. **Detect Cameras**: Find built-in or USB camera
3. **Train Model**: Use baby images (see Training tab)
4. **Configure Detection**:
   - Model: `./runs/train/best.pt`
   - Confidence: 0.30
   - Class: "baby"
   - Camera: 0 (built-in)
5. **Start Camera**: Monitor in real-time
6. **Adjust Settings**: Fine-tune as needed

### Desk Occupancy Monitor

1. **Position Camera**: Overlook desk area
2. **Use Pre-trained Model**: COCO model for people
3. **Configure Detection**:
   - Model: `yolov5s.pt`
   - Confidence: 0.40
   - Class: "person"
   - Camera: 1 (external USB)
4. **Start Camera**: Monitor occupancy
5. **Log Results**: Track presence over time

### Pet Detection

1. **Position Camera**: Cover pet area
2. **Train Custom Model**: Use pet images
3. **Configure Detection**:
   - Model: `./runs/train/pet_model.pt`
   - Confidence: 0.25
   - Classes: "dog,cat"
   - Camera: 0
4. **Start Camera**: Monitor pets
5. **Adjust Confidence**: Based on results

## Support

For issues or questions:
1. Check **Logs** tab for error messages
2. Try **Detect Cameras** to verify camera availability
3. Test camera in other apps (Windows Camera, VLC)
4. Review troubleshooting section above

## Quick Reference

### Common Camera Indices
- **0**: Built-in laptop webcam
- **1**: First external USB camera
- **2**: Second external USB camera

### Recommended Settings
- **Confidence**: 0.25 (adjust based on results)
- **Resolution**: 640x480 (default, good balance)
- **Model**: yolov5s (balanced speed/accuracy)

### Keyboard Shortcuts
- None currently (use mouse/buttons)

### Status Indicators
- **🔴 LIVE**: Camera is streaming
- **Camera stopped**: Camera is off
- **Opening camera...**: Connecting to camera
- **Error**: Camera error occurred

---

**Ready to start detecting?** Head to the **📹 USB Cam** tab and click 'Detect Cameras'! 🚀
