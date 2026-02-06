# YOLO Dataset Folder Structure

## Supported Structures

Our system supports the standard YOLO dataset structure with images and labels in the same folder.

## Standard Structure (What We Use)

```
dataset_root/
└── images/
    ├── train/
    │   ├── image001.jpg
    │   ├── image001.txt  ← Label file
    │   ├── image002.jpg
    │   ├── image002.txt  ← Label file
    │   └── ...
    ├── val/
    │   ├── image101.jpg
    │   ├── image101.txt
    │   └── ...
    └── test/
        ├── image201.jpg
        ├── image201.txt
        └── ...
```

## Key Points:

1. **Dataset Root**: Main folder (e.g., `./data/baby_dataset`)
2. **Images Folder**: Contains train/val/test subfolders
3. **Labels**: Stored in SAME folder as images (Yolo_Label standard)
4. **Naming**: Each image has a matching `.txt` file with same name

## Example:

If your dataset root is: `E:\test\Kiro_baby\data\baby_dataset`

Then your structure should be:
```
E:\test\Kiro_baby\data\baby_dataset/
└── images/
    ├── train/
    │   ├── baby001.jpg
    │   ├── baby001.txt
    │   ├── baby002.jpg
    │   └── baby002.txt
    ├── val/
    │   ├── baby101.jpg
    │   └── baby101.txt
    └── test/
        ├── baby201.jpg
        └── baby201.txt
```

## How to Set Up:

### Method 1: Using Labeling Tab
1. Put all your images in a folder (e.g., `E:\test\Kiro_baby\my_images\`)
2. Go to **Labeling** tab
3. Load images and label them
4. Labels are saved in the same folder as images
5. Later, use **Dataset** tab to organize into train/val/test

### Method 2: Manual Setup
1. Create the folder structure manually
2. Put images in `images/train/` folder
3. Put corresponding `.txt` labels in the same folder
4. Use **Dataset** tab → **Split Dataset** to create val/test splits

### Method 3: Using Dataset Tab
1. Put images and labels together in a source folder
2. Go to **Dataset** tab
3. Set "Dataset Root" to your target location
4. Set "Source Directory" to your source folder
5. Click "🚀 Collect Images & Labels"
6. Click "✂️ Split Dataset" to create train/val/test splits

## Label File Format

Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

Example (`baby001.txt`):
```
0 0.5 0.5 0.3 0.4
0 0.7 0.3 0.25 0.35
```

All coordinates are normalized (0.0 to 1.0).

## Validation

The **Dataset Statistics** feature checks:
- ✅ Each image has a corresponding label
- ✅ Each label has a corresponding image
- ✅ Label format is correct (5 values per line)
- ✅ Coordinates are normalized (0-1 range)
- ✅ At least some bounding boxes exist

## Common Issues:

### Issue: "Images without Labels"
**Problem**: `.txt` files are missing

**Solution**: 
- Label your images using the Labeling tab
- Or manually create `.txt` files for each image

### Issue: "Labels without Images"
**Problem**: Orphaned `.txt` files

**Solution**:
- Delete unused `.txt` files
- Or add corresponding images

### Issue: "Not Ready for Training"
**Problem**: Dataset validation failed

**Solution**:
- Check the issues list in Dataset Statistics
- Fix each issue (missing labels, wrong format, etc.)
- Refresh statistics to verify

## Created by Andy Kong
