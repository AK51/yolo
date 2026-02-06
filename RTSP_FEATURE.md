# 📡 RTSP Real-Time Detection Feature

## Overview

The RTSP Real-Time Detection feature allows you to connect to network cameras and perform live object detection on video streams. This is perfect for surveillance, monitoring, and real-time analysis applications.

## Features

✅ **Real-Time Detection**: Process live video streams with YOLO object detection  
✅ **Network Camera Support**: Connect to any RTSP-compatible camera  
✅ **Live Visualization**: See bounding boxes and labels in real-time  
✅ **Performance Metrics**: Monitor FPS and detection count  
✅ **Configurable Confidence**: Adjust detection sensitivity on the fly  
✅ **Multi-Class Support**: Detect multiple object classes simultaneously  

## How to Use

### 1. Prepare Your Model

Before using RTSP detection, you need a trained YOLO model:

1. Train a model using the **🚀 Training** tab
2. Or use a pre-trained model (e.g., `./runs/train/best.pt`)
3. Note the model path for later

### 2. Set Up Your RTSP Camera

Ensure your network camera is:
- Connected to the same network as your computer
- Configured with RTSP streaming enabled
- Accessible via its RTSP URL

**Common RTSP URL formats:**
```
rtsp://192.168.1.100:554/stream
rtsp://username:password@192.168.1.100:554/stream
rtsp://admin:admin123@camera.local:554/h264
```

### 3. Configure Detection Settings

In the **📡 RTSP** tab:

1. **Model Path**: Browse and select your trained model
2. **Confidence**: Set detection threshold (0.25 default)
   - Lower (0.15): More detections, more false positives
   - Higher (0.5): Fewer detections, higher accuracy
3. **Class Names**: Enter comma-separated class names (e.g., "baby,person,dog")

### 4. Enter RTSP URL

In the **RTSP URL** field, enter your camera's stream URL:
```
rtsp://username:password@192.168.1.100:554/stream
```

**Important**: Replace with your actual camera credentials and IP address.

### 5. Start Streaming

1. Click **▶️ Start Stream**
2. Wait for connection (usually 2-5 seconds)
3. Watch live detections appear on screen!

### 6. Monitor Performance

The interface shows:
- **🔴 LIVE**: Stream status indicator
- **Detections**: Number of objects detected in current frame
- **FPS**: Frames processed per second

### 7. Stop Streaming

Click **⏹️ Stop Stream** when finished.

## RTSP URL Examples

### Generic IP Camera
```
rtsp://192.168.1.100:554/stream
```

### Camera with Authentication
```
rtsp://admin:password123@192.168.1.100:554/stream
```

### Hikvision Camera
```
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
```

### Dahua Camera
```
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

### Axis Camera
```
rtsp://root:password@192.168.1.100/axis-media/media.amp
```

### Reolink Camera
```
rtsp://admin:password@192.168.1.100:554/h264Preview_01_main
```

## Performance Tips

### For Better FPS:
- Use GPU (CUDA) if available
- Use smaller model (yolov5n)
- Reduce camera resolution
- Lower confidence threshold slightly

### For Better Accuracy:
- Use larger model (yolov5s or yolov5m)
- Increase confidence threshold
- Ensure good lighting at camera
- Train model with similar scenes

### For Lower Bandwidth:
- Reduce camera stream quality
- Use H.264 compression
- Lower frame rate at camera
- Use local network (not internet)

## Troubleshooting

### "Failed to connect to RTSP stream"

**Possible causes:**
1. Wrong RTSP URL
2. Camera not accessible on network
3. Incorrect username/password
4. Firewall blocking connection
5. Camera RTSP disabled

**Solutions:**
- Verify URL in VLC Media Player first
- Check camera IP address (ping it)
- Confirm credentials are correct
- Disable firewall temporarily to test
- Enable RTSP in camera settings

### "Lost connection to stream"

**Possible causes:**
1. Network interruption
2. Camera restarted
3. Bandwidth issues
4. Camera timeout

**Solutions:**
- Check network stability
- Restart camera
- Reduce stream quality
- Increase camera timeout settings

### Low FPS / Laggy Stream

**Possible causes:**
1. CPU/GPU overload
2. High resolution stream
3. Large model (yolov5l/x)
4. Network bandwidth issues

**Solutions:**
- Use smaller model (yolov5n)
- Reduce camera resolution
- Close other applications
- Use wired connection instead of WiFi

### No Detections Appearing

**Possible causes:**
1. Confidence threshold too high
2. Model not trained for objects in view
3. Poor lighting conditions
4. Objects too small/far

**Solutions:**
- Lower confidence to 0.15-0.20
- Retrain model with similar scenes
- Improve camera lighting
- Move camera closer to objects

## Use Cases

### 🏠 Home Security
- Monitor entrances for people
- Detect package deliveries
- Pet monitoring
- Baby monitoring

### 🏢 Business Applications
- Customer counting
- Queue management
- Safety compliance monitoring
- Inventory tracking

### 🚗 Traffic Monitoring
- Vehicle counting
- License plate detection
- Parking space monitoring
- Traffic flow analysis

### 🏥 Healthcare
- Patient monitoring
- Fall detection
- PPE compliance
- Occupancy monitoring

## Technical Details

### Stream Processing
- **Backend**: OpenCV VideoCapture
- **Detection**: Ultralytics YOLO
- **Threading**: PyQt5 QThread for non-blocking operation
- **Frame Rate**: Depends on model and hardware (typically 10-30 FPS)

### Supported Protocols
- RTSP (Real-Time Streaming Protocol)
- RTSP over TCP
- RTSP over UDP

### Supported Codecs
- H.264 (most common)
- H.265 (HEVC)
- MJPEG

### Network Requirements
- Local network: 1-5 Mbps per stream
- Internet: 5-10 Mbps per stream
- Latency: < 500ms recommended

## Best Practices

### Security
1. **Use strong passwords** for camera credentials
2. **Keep cameras on isolated network** (VLAN)
3. **Don't expose RTSP to internet** without VPN
4. **Update camera firmware** regularly

### Performance
1. **Test with VLC first** to verify stream works
2. **Start with low confidence** (0.20) and adjust up
3. **Monitor system resources** (CPU/GPU/RAM)
4. **Use wired connection** for stability

### Reliability
1. **Set up camera watchdog** for auto-restart
2. **Configure camera keep-alive** settings
3. **Monitor connection status** in logs
4. **Have backup power** for cameras

## Advanced Configuration

### Multiple Cameras
To monitor multiple cameras:
1. Open multiple instances of the GUI
2. Or modify code to support multiple streams
3. Use different ports for each camera

### Recording Detections
To save detected frames:
1. Modify RTSPThread to save frames
2. Add recording button to UI
3. Store frames with timestamps

### Custom Alerts
To trigger alerts on detection:
1. Add alert logic to RTSPThread
2. Send notifications (email, SMS, etc.)
3. Log events to database

## Limitations

- **Single Stream**: Currently supports one stream at a time
- **No Recording**: Live view only (no built-in recording)
- **No Audio**: Video only, no audio processing
- **Network Dependent**: Requires stable network connection

## Future Enhancements

Potential improvements:
- Multi-stream support
- Built-in recording functionality
- Motion detection zones
- Alert system integration
- Cloud storage support
- Mobile app companion

## Support

For issues or questions:
1. Check **Logs** tab for error messages
2. Test RTSP URL in VLC Media Player
3. Verify camera network connectivity
4. Review troubleshooting section above

## Example Workflow

### Baby Monitor Setup

1. **Position Camera**: Point at crib/play area
2. **Get RTSP URL**: From camera settings
3. **Train Model**: Use baby images (see Training tab)
4. **Configure Detection**:
   - Model: `./runs/train/best.pt`
   - Confidence: 0.30
   - Class: "baby"
5. **Start Stream**: Monitor in real-time
6. **Adjust Settings**: Fine-tune confidence as needed

### Security Camera Setup

1. **Position Camera**: Cover entrance/area
2. **Get RTSP URL**: From camera admin panel
3. **Use Pre-trained Model**: COCO model for people
4. **Configure Detection**:
   - Model: `yolov5s.pt`
   - Confidence: 0.40
   - Classes: "person,car,dog"
5. **Start Stream**: Monitor activity
6. **Review Logs**: Check detection events

---

**Ready to monitor in real-time?** Head to the **📡 RTSP** tab and start streaming! 🚀
