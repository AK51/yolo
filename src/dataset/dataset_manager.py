"""Dataset Manager for YOLO training pipeline"""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from src.models import (
    CollectionResult,
    DatasetConfig,
    DatasetManifest,
    DatasetStatistics,
    ImageInfo,
    SplitInfo,
    ValidationReport,
)


class DatasetManager:
    """Manages dataset collection, organization, and validation"""
    
    def __init__(self, dataset_root: Path, config: Optional[DatasetConfig] = None):
        """
        Initialize dataset manager with root directory and configuration.
        
        Args:
            dataset_root: Root directory for dataset storage
            config: Dataset configuration (uses defaults if None)
        """
        self.dataset_root = Path(dataset_root)
        self.config = config or DatasetConfig()
        self.manifest_path = self.dataset_root / "manifest.json"
        self._image_hashes: Dict[str, str] = {}  # hash -> image_id mapping
        
        # Initialize directory structure
        self._setup_directories()
        
        # Load or initialize manifest
        self.manifest = self._load_or_create_manifest()
    
    def _setup_directories(self) -> None:
        """Create standardized directory structure"""
        directories = [
            self.dataset_root / "images" / "train",
            self.dataset_root / "images" / "val",
            self.dataset_root / "images" / "test",
            self.dataset_root / "labels" / "train",
            self.dataset_root / "labels" / "val",
            self.dataset_root / "labels" / "test",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_or_create_manifest(self) -> DatasetManifest:
        """Load existing manifest or create new one"""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                data = json.load(f)
                # Convert datetime string back to datetime
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                
                # Convert images list from dicts to ImageInfo objects
                if 'images' in data and data['images']:
                    data['images'] = [ImageInfo(**img) if isinstance(img, dict) else img 
                                     for img in data['images']]
                
                # Convert classes list from dicts to ClassInfo objects if present
                if 'classes' in data and data['classes']:
                    from src.models import ClassInfo
                    data['classes'] = [ClassInfo(**cls) if isinstance(cls, dict) else cls 
                                      for cls in data['classes']]
                
                # Convert splits from dict to SplitInfo object if present
                if 'splits' in data and data['splits'] and isinstance(data['splits'], dict):
                    from src.models import SplitInfo
                    data['splits'] = SplitInfo(**data['splits'])
                
                return DatasetManifest(**data)
        else:
            manifest = DatasetManifest(
                version="1.0",
                created_at=datetime.now(),
                dataset_name=self.dataset_root.name,
                total_images=0,
                total_annotations=0,
                classes=[],
                images=[],
                splits=None
            )
            # Save the new manifest to disk
            self.manifest = manifest
            self._save_manifest()
            return manifest
    
    def _save_manifest(self) -> None:
        """Save manifest to disk"""
        # Convert to dict and handle datetime serialization
        manifest_dict = {
            'version': self.manifest.version,
            'created_at': self.manifest.created_at.isoformat(),
            'dataset_name': self.manifest.dataset_name,
            'total_images': self.manifest.total_images,
            'total_annotations': self.manifest.total_annotations,
            'classes': [vars(c) for c in self.manifest.classes],
            'images': [vars(img) for img in self.manifest.images],
            'splits': vars(self.manifest.splits) if self.manifest.splits else None
        }
        
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest_dict, f, indent=2)
    
    def _compute_image_hash(self, image_path: Path) -> str:
        """Compute hash of image content for duplicate detection"""
        hasher = hashlib.md5()
        with open(image_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    
    def _validate_image(self, image_path: Path) -> ValidationReport:
        """
        Validate image format and dimensions.
        
        Args:
            image_path: Path to image file
            
        Returns:
            ValidationReport with validation results
        """
        errors = []
        warnings = []
        
        # Check file extension
        if image_path.suffix.lower() not in self.config.supported_formats:
            errors.append(f"Unsupported format: {image_path.suffix}. "
                         f"Supported: {self.config.supported_formats}")
            return ValidationReport(is_valid=False, errors=errors, warnings=warnings, total_checked=1)
        
        # Try to open and validate image
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Check dimensions
                if width < self.config.min_image_width:
                    errors.append(f"Image width {width} below minimum {self.config.min_image_width}")
                
                if height < self.config.min_image_height:
                    errors.append(f"Image height {height} below minimum {self.config.min_image_height}")
                
        except Exception as e:
            errors.append(f"Failed to open image: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationReport(is_valid=is_valid, errors=errors, warnings=warnings, total_checked=1)
    
    def import_local_images(self, source_dir: Path, copy: bool = True, label_dir: Optional[Path] = None) -> CollectionResult:
        """
        Import images and labels from local directory into dataset.
        
        For object detection, expects:
        - Images in source_dir/images/ or source_dir/
        - Labels in source_dir/labels/ or label_dir/ (YOLO format .txt files)
        
        Args:
            source_dir: Source directory containing images (and optionally labels)
            copy: If True, copy files; if False, move files
            label_dir: Optional separate directory containing labels (for parallel structure)
            
        Returns:
            CollectionResult with import statistics
        """
        source_dir = Path(source_dir)
        images_collected = 0
        images_failed = 0
        labels_collected = 0
        total_size = 0
        errors = []
        warnings = []
        
        if not source_dir.exists():
            errors.append(f"Source directory does not exist: {source_dir}")
            return CollectionResult(
                images_collected=0,
                images_failed=0,
                total_size_bytes=0,
                source_attribution={'source': str(source_dir)},
                errors=errors
            )
        
        # Check for images in subdirectory or root
        images_source = source_dir / "images" if (source_dir / "images").exists() else source_dir
        
        # Determine labels source
        if label_dir:
            # User specified a separate label directory
            label_dir_path = Path(label_dir)
            # Check if label_dir is the same as images_source (same folder structure)
            if label_dir_path.resolve() == images_source.resolve():
                # Labels are in the same folder as images
                labels_source = None
            else:
                # Labels are in a separate folder
                labels_source = label_dir_path
        elif (source_dir / "labels").exists():
            # Labels in parallel folder structure
            labels_source = source_dir / "labels"
        else:
            # Labels might be in same folder as images (will check per image)
            labels_source = None
        
        # Find all image files (use set to avoid case-insensitive duplicates on Windows)
        image_files_set = set()
        for ext in self.config.supported_formats:
            for img_path in images_source.glob(f"*{ext}"):
                image_files_set.add(img_path)
            for img_path in images_source.glob(f"*{ext.upper()}"):
                image_files_set.add(img_path)
        
        image_files = list(image_files_set)
        
        for image_path in image_files:
            try:
                # Validate image
                validation = self._validate_image(image_path)
                if not validation.is_valid:
                    images_failed += 1
                    errors.extend(validation.errors)
                    continue
                
                # Check for duplicates
                if self.config.check_duplicates:
                    img_hash = self._compute_image_hash(image_path)
                    if img_hash in self._image_hashes:
                        images_failed += 1
                        errors.append(f"Duplicate image: {image_path.name}")
                        continue
                    self._image_hashes[img_hash] = image_path.stem
                
                # Get image dimensions
                with Image.open(image_path) as img:
                    width, height = img.size
                
                # Copy/move image to dataset
                dest_image_path = self.dataset_root / "images" / "train" / image_path.name
                if copy:
                    shutil.copy2(image_path, dest_image_path)
                else:
                    shutil.move(str(image_path), dest_image_path)
                
                # Check for corresponding label file
                annotation_count = 0
                label_found = False
                
                # Try labels_source first (parallel structure or user-specified)
                if labels_source:
                    label_path = labels_source / f"{image_path.stem}.txt"
                    if label_path.exists():
                        dest_label_path = self.dataset_root / "labels" / "train" / f"{image_path.stem}.txt"
                        if copy:
                            shutil.copy2(label_path, dest_label_path)
                        else:
                            shutil.move(str(label_path), dest_label_path)
                        labels_collected += 1
                        label_found = True
                        
                        # Count annotations in label file
                        with open(label_path, 'r') as f:
                            annotation_count = len([line for line in f if line.strip()])
                
                # If not found in labels_source, try same folder as image
                if not label_found:
                    label_path = image_path.parent / f"{image_path.stem}.txt"
                    if label_path.exists():
                        dest_label_path = self.dataset_root / "labels" / "train" / f"{image_path.stem}.txt"
                        if copy:
                            shutil.copy2(label_path, dest_label_path)
                        else:
                            shutil.move(str(label_path), dest_label_path)
                        labels_collected += 1
                        label_found = True
                        
                        # Count annotations in label file
                        with open(label_path, 'r') as f:
                            annotation_count = len([line for line in f if line.strip()])
                
                # Warn if no label found
                if not label_found:
                    warnings.append(f"No label file found for {image_path.name}")
                
                # Add to manifest
                image_info = ImageInfo(
                    image_id=image_path.stem,
                    filename=image_path.name,
                    width=width,
                    height=height,
                    source=str(source_dir),
                    license="local",
                    attribution="local_import",
                    split="train",
                    annotation_count=annotation_count
                )
                self.manifest.images.append(image_info)
                
                images_collected += 1
                total_size += image_path.stat().st_size
                
            except Exception as e:
                images_failed += 1
                errors.append(f"Failed to import {image_path.name}: {str(e)}")
        
        # Update manifest
        self.manifest.total_images = len(self.manifest.images)
        # Safely sum annotation counts, handling both ImageInfo objects and dicts
        total_annotations = 0
        for img in self.manifest.images:
            if isinstance(img, dict):
                total_annotations += img.get('annotation_count', 0)
            else:
                total_annotations += img.annotation_count
        self.manifest.total_annotations = total_annotations
        self._save_manifest()
        
        # Add summary to errors if there are warnings
        if warnings:
            errors.append(f"⚠️ Collected {images_collected} images but only {labels_collected} labels. "
                         f"For object detection, each image needs a corresponding .txt label file.")
        
        return CollectionResult(
            images_collected=images_collected,
            images_failed=images_failed,
            total_size_bytes=total_size,
            source_attribution={
                'source': str(source_dir),
                'type': 'local',
                'labels_collected': labels_collected,
                'images_without_labels': images_collected - labels_collected
            },
            errors=errors
        )
    
    def validate_dataset(self) -> ValidationReport:
        """
        Validate all images in dataset.
        
        Returns:
            ValidationReport with validation results
        """
        errors = []
        warnings = []
        total_checked = 0
        
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_root / "images" / split
            if not images_dir.exists():
                continue
            
            for image_path in images_dir.iterdir():
                if image_path.is_file():
                    total_checked += 1
                    validation = self._validate_image(image_path)
                    errors.extend(validation.errors)
                    warnings.extend(validation.warnings)
        
        is_valid = len(errors) == 0
        return ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            total_checked=total_checked
        )
    
    def get_statistics(self) -> DatasetStatistics:
        """
        Get dataset statistics by counting actual files in directories.
        
        Returns:
            DatasetStatistics with counts and distribution
        """
        total_size = 0
        class_dist = {}
        train_count = 0
        val_count = 0
        test_count = 0
        
        # Count actual images in each split directory
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_root / "images" / split
            if images_dir.exists():
                image_files = []
                for ext in self.config.supported_formats:
                    image_files.extend(list(images_dir.glob(f"*{ext}")))
                    image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
                
                # Remove duplicates (case-insensitive on Windows)
                unique_files = {f.name.lower(): f for f in image_files}
                count = len(unique_files)
                
                if split == 'train':
                    train_count = count
                elif split == 'val':
                    val_count = count
                elif split == 'test':
                    test_count = count
                
                # Calculate total size
                for img_path in unique_files.values():
                    total_size += img_path.stat().st_size
        
        # Calculate total images
        total_images = train_count + val_count + test_count
        
        # Get class distribution from manifest if available
        for class_info in self.manifest.classes:
            class_dist[class_info.class_name] = class_info.instance_count
        
        return DatasetStatistics(
            total_images=total_images,
            total_size_bytes=total_size,
            class_distribution=class_dist,
            train_count=train_count,
            val_count=val_count,
            test_count=test_count
        )
    
    def export_manifest(self, output_path: Path) -> None:
        """
        Export dataset manifest to specified path.
        
        Args:
            output_path: Path to export manifest
        """
        output_path = Path(output_path)
        shutil.copy2(self.manifest_path, output_path)
    
    def split_dataset(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42
    ) -> Dict[str, int]:
        """
        Split dataset into train/val/test sets with stratification.
        Skips duplicate filenames to avoid conflicts.
        
        Args:
            train_ratio: Ratio for training set (0-1)
            val_ratio: Ratio for validation set (0-1)
            test_ratio: Ratio for test set (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with split counts
        """
        import random
        random.seed(seed)
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Get all images currently in train (unsplit)
        train_dir = self.dataset_root / "images" / "train"
        all_images = list(train_dir.glob("*"))
        all_images = [img for img in all_images if img.suffix.lower() in self.config.supported_formats]
        
        # Remove duplicates by filename (keep first occurrence)
        seen_names = set()
        unique_images = []
        duplicates_skipped = 0
        
        for img in all_images:
            if img.name not in seen_names:
                seen_names.add(img.name)
                unique_images.append(img)
            else:
                duplicates_skipped += 1
        
        all_images = unique_images
        
        # Shuffle for random split
        random.shuffle(all_images)
        
        total_images = len(all_images)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        test_count = total_images - train_count - val_count
        
        # Split images
        train_images = all_images[:train_count]
        val_images = all_images[train_count:train_count + val_count]
        test_images = all_images[train_count + val_count:]
        
        # Move images to respective directories
        for img in val_images:
            dest = self.dataset_root / "images" / "val" / img.name
            # Skip if destination already exists (duplicate)
            if dest.exists():
                continue
            shutil.move(str(img), dest)
            
            # Move corresponding label if exists
            label_src = self.dataset_root / "labels" / "train" / f"{img.stem}.txt"
            if label_src.exists():
                label_dest = self.dataset_root / "labels" / "val" / f"{img.stem}.txt"
                if not label_dest.exists():  # Skip if label already exists
                    shutil.move(str(label_src), label_dest)
        
        for img in test_images:
            dest = self.dataset_root / "images" / "test" / img.name
            # Skip if destination already exists (duplicate)
            if dest.exists():
                continue
            shutil.move(str(img), dest)
            
            # Move corresponding label if exists
            label_src = self.dataset_root / "labels" / "train" / f"{img.stem}.txt"
            if label_src.exists():
                label_dest = self.dataset_root / "labels" / "test" / f"{img.stem}.txt"
                if not label_dest.exists():  # Skip if label already exists
                    shutil.move(str(label_src), label_dest)
        
        # Update manifest
        for image_info in self.manifest.images:
            if image_info.filename in [img.name for img in val_images]:
                image_info.split = "val"
            elif image_info.filename in [img.name for img in test_images]:
                image_info.split = "test"
        
        # Update split info in manifest
        self.manifest.splits = SplitInfo(
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
            split_ratios=(train_ratio, val_ratio, test_ratio),
            random_seed=seed
        )
        self._save_manifest()
        
        result = {
            'train': train_count,
            'val': val_count,
            'test': test_count
        }
        
        if duplicates_skipped > 0:
            result['duplicates_skipped'] = duplicates_skipped
        
        return result
    
    def collect_from_coco(self, categories: List[str], max_images: int) -> CollectionResult:
        """
        Download images from COCO dataset for specified categories.
        
        Args:
            categories: List of category names to collect
            max_images: Maximum number of images to download
            
        Returns:
            CollectionResult with collection statistics
        """
        # Placeholder implementation - requires pycocotools
        errors = ["COCO collection not yet implemented. Install pycocotools and implement."]
        return CollectionResult(
            images_collected=0,
            images_failed=0,
            total_size_bytes=0,
            source_attribution={'source': 'COCO', 'categories': categories},
            errors=errors
        )
    
    def collect_from_open_images(self, categories: List[str], max_images: int) -> CollectionResult:
        """
        Download images from Open Images dataset for specified categories.
        
        Args:
            categories: List of category names to collect
            max_images: Maximum number of images to download
            
        Returns:
            CollectionResult with collection statistics
        """
        # Placeholder implementation
        errors = ["Open Images collection not yet implemented."]
        return CollectionResult(
            images_collected=0,
            images_failed=0,
            total_size_bytes=0,
            source_attribution={'source': 'Open Images', 'categories': categories},
            errors=errors
        )

    def export_license_summary(self, output_path: Optional[Path] = None) -> Path:
        """
        Export license summary file with all image licenses.
        
        Args:
            output_path: Path to export license summary (defaults to dataset_root/licenses.txt)
            
        Returns:
            Path to the exported license summary file
        """
        if output_path is None:
            output_path = self.dataset_root / "licenses.txt"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DATASET LICENSE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Dataset: {self.manifest.dataset_name}\n")
            f.write(f"Total Images: {self.manifest.total_images}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group images by license type
            license_groups: Dict[str, List[ImageInfo]] = {}
            for image_info in self.manifest.images:
                license_type = image_info.license or "Unknown"
                if license_type not in license_groups:
                    license_groups[license_type] = []
                license_groups[license_type].append(image_info)
            
            # Write license groups
            for license_type, images in sorted(license_groups.items()):
                f.write("-" * 80 + "\n")
                f.write(f"License: {license_type}\n")
                f.write(f"Image Count: {len(images)}\n")
                f.write("-" * 80 + "\n\n")
                
                for img in images:
                    f.write(f"  File: {img.filename}\n")
                    f.write(f"  Source: {img.source}\n")
                    if img.attribution:
                        f.write(f"  Attribution: {img.attribution}\n")
                    f.write("\n")
                
                f.write("\n")
        
        return output_path
    
    def filter_images_by_license(self, license_type: str) -> List[ImageInfo]:
        """
        Filter images by license type.
        
        Args:
            license_type: License type to filter by
            
        Returns:
            List of ImageInfo objects with matching license
        """
        return [img for img in self.manifest.images if img.license == license_type]
    
    def check_license_compatibility(self) -> List[str]:
        """
        Check for incompatible licenses in the dataset.
        
        Returns:
            List of warning messages about license incompatibilities
        """
        warnings = []
        
        # Get all unique licenses
        licenses = set(img.license for img in self.manifest.images if img.license)
        
        # Define incompatible license combinations
        restrictive_licenses = {'CC-BY-NC', 'CC-BY-NC-SA', 'CC-BY-NC-ND'}
        commercial_licenses = {'CC-BY', 'CC-BY-SA', 'CC0', 'MIT', 'Apache-2.0'}
        
        # Check for mixing restrictive and commercial licenses
        has_restrictive = any(lic in restrictive_licenses for lic in licenses)
        has_commercial = any(lic in commercial_licenses for lic in licenses)
        
        if has_restrictive and has_commercial:
            warnings.append(
                "Warning: Dataset contains both non-commercial (NC) and commercial-use "
                "licenses. This may restrict commercial use of the entire dataset."
            )
        
        # Check for NoDerivatives licenses
        no_derivatives = [lic for lic in licenses if 'ND' in lic]
        if no_derivatives:
            warnings.append(
                f"Warning: Dataset contains NoDerivatives licenses ({', '.join(no_derivatives)}). "
                "This may restrict modification and derivative works."
            )
        
        # Check for ShareAlike requirements
        share_alike = [lic for lic in licenses if 'SA' in lic]
        if share_alike and len(licenses) > 1:
            warnings.append(
                f"Warning: Dataset contains ShareAlike licenses ({', '.join(share_alike)}). "
                "Derivative works may need to be shared under the same license."
            )
        
        return warnings
    
    def get_license_statistics(self) -> Dict[str, int]:
        """
        Get statistics on license distribution in the dataset.
        
        Returns:
            Dictionary mapping license types to image counts
        """
        license_counts: Dict[str, int] = {}
        
        for img in self.manifest.images:
            license_type = img.license or "Unknown"
            license_counts[license_type] = license_counts.get(license_type, 0) + 1
        
        return license_counts
