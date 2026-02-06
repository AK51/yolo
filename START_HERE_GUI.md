# 🚀 START HERE - Launch the GUI

## Quick Start

**Double-click this file to launch the GUI:**

```
launch_gui_venv.bat
```

## ⚠️ Important

- ✅ **USE:** `launch_gui_venv.bat` (Virtual Environment - GPU Working)
- ❌ **DON'T USE:** `launch_gui.bat` (System Python - Has DLL Errors)

## Why?

Your system Python has PyTorch 2.10.0+cu128 which has a **DLL initialization bug**.

Your virtual environment has PyTorch 2.1.0+cu121 which **works correctly with GPU**.

## Verification

After launching, you should see:

```
========================================
  YOLO Training Pipeline
  Virtual Environment Mode (GPU)
========================================

Python location:
   E:\test\Kiro_baby\.venv\Scripts\python.exe

Checking PyTorch installation...
  PyTorch version: 2.1.0+cu121
  CUDA available: True
```

## If You Get DLL Errors

You're using the wrong launcher! Close the GUI and use `launch_gui_venv.bat` instead.

---

**For more details, see:** `HOW_TO_LAUNCH_GUI.md`
