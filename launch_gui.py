"""Launch the YOLO Training Pipeline GUI"""
import sys
from pathlib import Path

# Ensure we're in the project directory
import os
os.chdir(Path(__file__).parent)

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 YOLO Training Pipeline - Starting...")
    print("=" * 60)
    print()
    
    print("🚀 Launching GUI...")
    print("=" * 60)
    print()
    
    from src.gui.main_window import main
    main()


