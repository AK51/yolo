"""Annotation Processor for format conversion and validation"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from src.models import (
    BoundingBox,
    ConversionResult,
    NormalizedBBox,
    ValidationReport,
)


class AnnotationProcessor:
    """Handles annotation format conversions and validation"""
    
    def __init__(self, class_mapping: Dict[int, str]):
        """
        Initialize with class ID to name mapping.
        
        Args:
            class_mapping: Dictionary mapping class IDs to class names
        """
        self.class_mapping = class_mapping
        self.reverse_mapping = {v: k for k, v in class_mapping.items()}
    
    def normalize_bbox(
        self,
        bbox: BoundingBox,
        image_width: int,
        image_height: int
    ) -> NormalizedBBox:
        """
        Normalize bounding box coordinates to YOLO format (0-1 range).
        
        Args:
            bbox: Bounding box in absolute coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            NormalizedBBox in YOLO format
        """
        # Calculate center and dimensions
        x_center = (bbox.x_min + bbox.x_max) / 2.0
        y_center = (bbox.y_min + bbox.y_max) / 2.0
        width = bbox.x_max - bbox.x_min
        height = bbox.y_max - bbox.y_min
        
        # Normalize to 0-1 range
        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = width / image_width
        height_norm = height / image_height
        
        return NormalizedBBox(
            x_center=x_center_norm,
            y_center=y_center_norm,
            width=width_norm,
            height=height_norm,
            class_id=bbox.class_id
        )
    
    def _validate_normalized_bbox(self, bbox: NormalizedBBox) -> List[str]:
        """Validate normalized bounding box coordinates"""
        errors = []
        
        # Check if coordinates are in valid range
        if not (0 <= bbox.x_center <= 1):
            errors.append(f"x_center {bbox.x_center} out of range [0, 1]")
        if not (0 <= bbox.y_center <= 1):
            errors.append(f"y_center {bbox.y_center} out of range [0, 1]")
        if not (0 <= bbox.width <= 1):
            errors.append(f"width {bbox.width} out of range [0, 1]")
        if not (0 <= bbox.height <= 1):
            errors.append(f"height {bbox.height} out of range [0, 1]")
        
        # Check if bbox extends beyond image boundaries
        if bbox.x_center - bbox.width / 2 < 0:
            errors.append("Bounding box extends beyond left image boundary")
        if bbox.x_center + bbox.width / 2 > 1:
            errors.append("Bounding box extends beyond right image boundary")
        if bbox.y_center - bbox.height / 2 < 0:
            errors.append("Bounding box extends beyond top image boundary")
        if bbox.y_center + bbox.height / 2 > 1:
            errors.append("Bounding box extends beyond bottom image boundary")
        
        # Check for negative dimensions
        if bbox.width <= 0:
            errors.append(f"Invalid width: {bbox.width}")
        if bbox.height <= 0:
            errors.append(f"Invalid height: {bbox.height}")
        
        return errors
    
    def convert_coco_to_yolo(
        self,
        coco_json: Path,
        images_dir: Path,
        output_dir: Path
    ) -> ConversionResult:
        """
        Convert COCO format annotations to YOLO format.
        
        Args:
            coco_json: Path to COCO JSON file
            images_dir: Directory containing images
            output_dir: Output directory for YOLO labels
            
        Returns:
            ConversionResult with conversion statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        annotations_converted = 0
        annotations_failed = 0
        images_processed = 0
        errors = []
        
        try:
            with open(coco_json, 'r') as f:
                coco_data = json.load(f)
            
            # Build image id to filename mapping
            image_map = {img['id']: img for img in coco_data.get('images', [])}
            
            # Group annotations by image
            annotations_by_image = {}
            for ann in coco_data.get('annotations', []):
                image_id = ann['image_id']
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(ann)
            
            # Process each image
            for image_id, annotations in annotations_by_image.items():
                if image_id not in image_map:
                    annotations_failed += len(annotations)
                    errors.append(f"Image ID {image_id} not found in images list")
                    continue
                
                image_info = image_map[image_id]
                image_width = image_info['width']
                image_height = image_info['height']
                filename = Path(image_info['file_name']).stem
                
                yolo_annotations = []
                for ann in annotations:
                    try:
                        # COCO bbox format: [x, y, width, height]
                        x, y, w, h = ann['bbox']
                        category_id = ann['category_id']
                        
                        # Convert to absolute coordinates
                        bbox = BoundingBox(
                            x_min=x,
                            y_min=y,
                            x_max=x + w,
                            y_max=y + h,
                            class_id=category_id
                        )
                        
                        # Normalize to YOLO format
                        norm_bbox = self.normalize_bbox(bbox, image_width, image_height)
                        
                        # Validate
                        bbox_errors = self._validate_normalized_bbox(norm_bbox)
                        if bbox_errors:
                            annotations_failed += 1
                            errors.extend(bbox_errors)
                            continue
                        
                        # Format: class_id x_center y_center width height
                        yolo_line = f"{norm_bbox.class_id} {norm_bbox.x_center:.6f} {norm_bbox.y_center:.6f} {norm_bbox.width:.6f} {norm_bbox.height:.6f}"
                        yolo_annotations.append(yolo_line)
                        annotations_converted += 1
                        
                    except Exception as e:
                        annotations_failed += 1
                        errors.append(f"Failed to convert annotation: {str(e)}")
                
                # Write YOLO label file
                if yolo_annotations:
                    label_path = output_dir / f"{filename}.txt"
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    images_processed += 1
                    
        except Exception as e:
            errors.append(f"Failed to process COCO JSON: {str(e)}")
        
        return ConversionResult(
            annotations_converted=annotations_converted,
            annotations_failed=annotations_failed,
            images_processed=images_processed,
            errors=errors
        )
    
    def convert_voc_to_yolo(
        self,
        voc_dir: Path,
        images_dir: Path,
        output_dir: Path
    ) -> ConversionResult:
        """
        Convert Pascal VOC format annotations to YOLO format.
        
        Args:
            voc_dir: Directory containing VOC XML files
            images_dir: Directory containing images
            output_dir: Output directory for YOLO labels
            
        Returns:
            ConversionResult with conversion statistics
        """
        voc_dir = Path(voc_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        annotations_converted = 0
        annotations_failed = 0
        images_processed = 0
        errors = []
        
        # Process each XML file
        for xml_file in voc_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get image dimensions
                size = root.find('size')
                image_width = int(size.find('width').text)
                image_height = int(size.find('height').text)
                
                yolo_annotations = []
                
                # Process each object
                for obj in root.findall('object'):
                    try:
                        class_name = obj.find('name').text
                        
                        # Get class ID from mapping
                        if class_name not in self.reverse_mapping:
                            annotations_failed += 1
                            errors.append(f"Unknown class: {class_name}")
                            continue
                        
                        class_id = self.reverse_mapping[class_name]
                        
                        # Get bounding box
                        bndbox = obj.find('bndbox')
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)
                        
                        bbox = BoundingBox(
                            x_min=xmin,
                            y_min=ymin,
                            x_max=xmax,
                            y_max=ymax,
                            class_id=class_id
                        )
                        
                        # Normalize to YOLO format
                        norm_bbox = self.normalize_bbox(bbox, image_width, image_height)
                        
                        # Validate
                        bbox_errors = self._validate_normalized_bbox(norm_bbox)
                        if bbox_errors:
                            annotations_failed += 1
                            errors.extend(bbox_errors)
                            continue
                        
                        yolo_line = f"{norm_bbox.class_id} {norm_bbox.x_center:.6f} {norm_bbox.y_center:.6f} {norm_bbox.width:.6f} {norm_bbox.height:.6f}"
                        yolo_annotations.append(yolo_line)
                        annotations_converted += 1
                        
                    except Exception as e:
                        annotations_failed += 1
                        errors.append(f"Failed to convert object in {xml_file.name}: {str(e)}")
                
                # Write YOLO label file
                if yolo_annotations:
                    label_path = output_dir / f"{xml_file.stem}.txt"
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    images_processed += 1
                    
            except Exception as e:
                errors.append(f"Failed to process {xml_file.name}: {str(e)}")
        
        return ConversionResult(
            annotations_converted=annotations_converted,
            annotations_failed=annotations_failed,
            images_processed=images_processed,
            errors=errors
        )
    
    def validate_yolo_annotations(
        self,
        labels_dir: Path,
        images_dir: Path
    ) -> ValidationReport:
        """
        Validate YOLO format annotations for correctness.
        
        Args:
            labels_dir: Directory containing YOLO label files
            images_dir: Directory containing corresponding images
            
        Returns:
            ValidationReport with validation results
        """
        labels_dir = Path(labels_dir)
        images_dir = Path(images_dir)
        
        errors = []
        warnings = []
        total_checked = 0
        
        # Check each label file
        for label_file in labels_dir.glob("*.txt"):
            total_checked += 1
            
            # Check if corresponding image exists
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_found = False
            for ext in image_extensions:
                image_path = images_dir / f"{label_file.stem}{ext}"
                if image_path.exists():
                    image_found = True
                    break
            
            if not image_found:
                errors.append(f"No image found for label file: {label_file.name}")
                continue
            
            # Validate label file content
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(f"{label_file.name} line {line_num}: Expected 5 values, got {len(parts)}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Create normalized bbox for validation
                        norm_bbox = NormalizedBBox(
                            x_center=x_center,
                            y_center=y_center,
                            width=width,
                            height=height,
                            class_id=class_id
                        )
                        
                        # Validate
                        bbox_errors = self._validate_normalized_bbox(norm_bbox)
                        if bbox_errors:
                            for err in bbox_errors:
                                errors.append(f"{label_file.name} line {line_num}: {err}")
                        
                        # Check class ID
                        if class_id not in self.class_mapping:
                            warnings.append(f"{label_file.name} line {line_num}: Unknown class ID {class_id}")
                        
                    except ValueError as e:
                        errors.append(f"{label_file.name} line {line_num}: Invalid number format - {str(e)}")
                        
            except Exception as e:
                errors.append(f"Failed to read {label_file.name}: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            total_checked=total_checked
        )
    
    def save_class_mapping(self, output_path: Path) -> None:
        """
        Save class mapping to file.
        
        Args:
            output_path: Path to save class mapping
        """
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(self.class_mapping, f, indent=2)
    
    def load_class_mapping(self, input_path: Path) -> None:
        """
        Load class mapping from file.
        
        Args:
            input_path: Path to class mapping file
        """
        input_path = Path(input_path)
        with open(input_path, 'r') as f:
            self.class_mapping = json.load(f)
        # Update reverse mapping
        self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}
