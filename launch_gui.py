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
    
    # Check for COCO8 dataset
    coco8_path = Path("./data/coco8")
    if not coco8_path.exists() or not (coco8_path / "images").exists():
        print("\n📥 COCO8 dataset not found. Downloading...")
        print("⏳ This may take a minute on first run...")
        
        try:
            from download_coco8 import download_coco8, verify_coco8_structure
            
            dataset_path = download_coco8()
            
            if dataset_path and verify_coco8_structure(dataset_path):
                print("\n✅ COCO8 dataset ready!")
            else:
                print("\n⚠️ Could not download COCO8 automatically")
                print("💡 The GUI will still start, but you may need to download manually")
        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}")
            print("💡 The GUI will still start")
    else:
        print("\n✅ COCO8 dataset found")
    
    print("\n🚀 Launching GUI...")
    print("=" * 60)
    print()
    
    from src.gui.main_window import main
    main()

