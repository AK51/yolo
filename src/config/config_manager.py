"""Configuration Manager for YOLO training"""
import yaml
from pathlib import Path
from typing import Optional

from src.models import TrainingConfig, ValidationReport


class ConfigurationManager:
    """Manages training configuration for YOLO models"""
    
    def __init__(self, yolo_version: str = "yolov5"):
        """
        Initialize for specific YOLO version.
        
        Args:
            yolo_version: YOLO version ('yolov5', 'yolov8', or 'yolo11')
        """
        if yolo_version not in ['yolov5', 'yolov8', 'yolo11']:
            raise ValueError(f"Unsupported YOLO version: {yolo_version}")
        
        self.yolo_version = yolo_version
    
    def load_config(self, config_path: Path) -> TrainingConfig:
        """
        Load and validate training configuration from file.
        
        Args:
            config_path: Path to configuration YAML file
            
        Returns:
            TrainingConfig object
        """
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create TrainingConfig from dict
        config = TrainingConfig(**config_dict)
        
        # Validate
        validation = self.validate_config(config)
        if not validation.is_valid:
            raise ValueError(f"Invalid configuration: {validation.errors}")
        
        return config
    
    def create_default_config(self, use_case: str = "general") -> TrainingConfig:
        """
        Create default configuration for common use case.
        
        Args:
            use_case: Use case ('general', 'small', 'large')
            
        Returns:
            TrainingConfig with default values
        """
        if use_case == "small":
            return TrainingConfig(
                model_architecture=f"{self.yolo_version}n",  # nano model
                epochs=50,
                batch_size=32,
                image_size=416,
                learning_rate=0.01,
                num_classes=1,
                class_names=["object"]
            )
        elif use_case == "large":
            return TrainingConfig(
                model_architecture=f"{self.yolo_version}x",  # extra large model
                epochs=300,
                batch_size=8,
                image_size=1280,
                learning_rate=0.001,
                num_classes=1,
                class_names=["object"]
            )
        else:  # general
            return TrainingConfig(
                model_architecture=f"{self.yolo_version}s",  # small model
                epochs=100,
                batch_size=16,
                image_size=640,
                learning_rate=0.01,
                num_classes=1,
                class_names=["object"]
            )
    
    def validate_config(self, config: TrainingConfig) -> ValidationReport:
        """
        Validate configuration parameters.
        
        Args:
            config: TrainingConfig to validate
            
        Returns:
            ValidationReport with validation results
        """
        errors = []
        warnings = []
        
        # Validate required parameters
        if not config.model_architecture:
            errors.append("model_architecture is required")
        
        if config.num_classes <= 0:
            errors.append(f"num_classes must be positive, got {config.num_classes}")
        
        if config.epochs <= 0:
            errors.append(f"epochs must be positive, got {config.epochs}")
        
        if config.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {config.batch_size}")
        
        if config.image_size <= 0:
            errors.append(f"image_size must be positive, got {config.image_size}")
        
        # Check if image size is divisible by 32 (YOLO requirement)
        if config.image_size % 32 != 0:
            errors.append(f"image_size must be divisible by 32, got {config.image_size}")
        
        if config.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {config.learning_rate}")
        
        # Validate model architecture
        valid_archs = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
                      'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                      'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']
        if config.model_architecture not in valid_archs:
            errors.append(f"Invalid model_architecture: {config.model_architecture}. "
                         f"Valid options: {valid_archs}")
        
        # Validate device
        if config.device not in ['cpu', 'cuda', 'mps']:
            warnings.append(f"Unusual device: {config.device}. Expected 'cpu', 'cuda', or 'mps'")
        
        # Check class names match num_classes
        if len(config.class_names) != config.num_classes:
            errors.append(f"Number of class_names ({len(config.class_names)}) "
                         f"doesn't match num_classes ({config.num_classes})")
        
        is_valid = len(errors) == 0
        return ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            total_checked=1
        )
    
    def export_yolo_yaml(self, config: TrainingConfig, output_path: Path) -> None:
        """
        Export configuration as YOLO-compatible YAML file.
        
        Args:
            config: TrainingConfig to export
            output_path: Path to save YAML file
        """
        output_path = Path(output_path)
        
        # Create YOLO-compatible config
        yolo_config = {
            'path': str(config.dataset_yaml.parent) if config.dataset_yaml else './data',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': config.num_classes,
            'names': config.class_names
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(yolo_config, f, default_flow_style=False)
