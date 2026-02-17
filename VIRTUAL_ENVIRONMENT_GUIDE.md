# Virtual Environment Quick Guide

## What is a Virtual Environment?

A virtual environment is an isolated Python installation that keeps your project dependencies separate from your system Python.

## Why Use It?

- ✅ Your virtual environment has PyTorch 2.1.0+cu121 (working)
- ❌ Your system Python has PyTorch 2.10.0+cu128 (DLL error)

## How to Activate

### PowerShell (Recommended)
```powershell
.\.venv\Scripts\Activate.ps1
```

### Command Prompt (CMD)
```cmd
.venv\Scripts\activate.bat
```

### Git Bash
```bash
source .venv/Scripts/activate
```

## How to Know It's Activated

You'll see `(.venv)` at the start of your prompt:

**Before activation:**
```
PS C:\path\to\yolo-training-pipeline>
```

**After activation:**
```
(.venv) PS C:\path\to\yolo-training-pipeline>
```

## Verify Correct Python

```powershell
python -c "import sys; print(sys.executable)"
```

**Should show:**
```
C:\path\to\yolo-training-pipeline\.venv\Scripts\python.exe
```

**Should NOT show:**
```
C:\Users\andyk\AppData\Local\Programs\Python\Python310\python.exe
```

## Verify PyTorch

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

**Expected output:**
```
PyTorch: 2.1.0+cu121
CUDA: True
```

## How to Deactivate

```powershell
deactivate
```

## Common Tasks

### Launch GUI (with venv)
```powershell
# Option 1: Use the launcher (automatically activates venv)
.\launch_gui_venv.bat

# Option 2: Activate manually, then launch
.\.venv\Scripts\Activate.ps1
python launch_gui.py
```

### Run Training Script
```powershell
.\.venv\Scripts\Activate.ps1
python test_real_training.py
```

### Install Packages
```powershell
.\.venv\Scripts\Activate.ps1
pip install package-name
```

### Update Requirements
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Troubleshooting

### PowerShell Execution Policy Error

If you get an error like:
```
cannot be loaded because running scripts is disabled on this system
```

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

### Wrong Python After Activation

If you still see system Python after activation:

1. **Close PowerShell completely**
2. **Open a new PowerShell window**
3. **Navigate to project directory**
4. **Activate again**

### Virtual Environment Not Found

If `.venv` doesn't exist:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Commands Cheat Sheet

| Task | Command |
|------|---------|
| Activate (PowerShell) | `.\.venv\Scripts\Activate.ps1` |
| Activate (CMD) | `.venv\Scripts\activate.bat` |
| Deactivate | `deactivate` |
| Check Python | `python -c "import sys; print(sys.executable)"` |
| Check PyTorch | `python -c "import torch; print(torch.__version__)"` |
| Launch GUI | `.\launch_gui_venv.bat` |
| Install packages | `pip install package-name` |

## Important Notes

1. **Always activate the venv before running Python commands**
2. **The `launch_gui_venv.bat` script activates it automatically**
3. **Each new terminal window needs activation**
4. **You can tell it's activated by the `(.venv)` prefix**

## Visual Guide

```
┌─────────────────────────────────────────┐
│  System Python (DON'T USE)              │
│  C:\Users\andyk\...\Python310\          │
│  PyTorch 2.10.0+cu128 ❌ DLL Error      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Virtual Environment (USE THIS!)        │
│  .venv\                                  │
│  PyTorch 2.1.0+cu121 ✅ Working         │
└─────────────────────────────────────────┘
```

## Summary

**To activate:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**To verify:**
```powershell
python -c "import sys; print(sys.executable)"
```

**Should see:**
```
C:\path\to\yolo-training-pipeline\.venv\Scripts\python.exe
```

---

**Created:** February 2, 2026
**Purpose:** Guide for activating and using the virtual environment
