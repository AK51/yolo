"""Core data models for YOLO training pipeline"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Dataset Manager Models
@dataclass
class DatasetConfig:
    """Configuration for dataset management"""
    min_image_width: int = 32
    min_image_height: int = 32
    supported_formats: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp'])
    check_duplicates: bool = True
    attribution_required: bool = True


@dataclass
class CollectionResult:
    """Result of dataset collection operation"""
    images_collected: int
    images_failed: int
    total_size_bytes: int
    source_attribution: Dict[str, str]
    errors: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a class in the dataset"""
    class_id: int
    class_name: str
    instance_count: int


@dataclass
class ImageInfo:
    """Information about an image in the dataset"""
    image_id: str
    filename: str
    width: int
    height: int
    source: str
    license: str
    attribution: str
    split: str  # 'train', 'val', or 'test'
    annotation_count: int


@dataclass
class SplitInfo:
    """Information about dataset splits"""
    train_count: int
    val_count: int
    test_count: int
    split_ratios: Tuple[float, float, float]
    random_seed: int


@dataclass
class DatasetManifest:
    """Complete dataset manifest"""
    version: str
    created_at: datetime
    dataset_name: str
    total_images: int
    total_annotations: int
    classes: List[ClassInfo]
    images: List[ImageInfo]
    splits: Optional[SplitInfo] = None


@dataclass
class DatasetStatistics:
    """Dataset statistics"""
    total_images: int
    total_size_bytes: int
    class_distribution: Dict[str, int]
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0


# Annotation Processor Models
@dataclass
class BoundingBox:
    """Bounding box in absolute coordinates"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_id: int


@dataclass
class NormalizedBBox:
    """Bounding box in YOLO normalized format"""
    x_center: float  # 0-1 normalized
    y_center: float  # 0-1 normalized
    width: float     # 0-1 normalized
    height: float    # 0-1 normalized
    class_id: int


@dataclass
class ConversionResult:
    """Result of annotation format conversion"""
    annotations_converted: int
    annotations_failed: int
    images_processed: int
    errors: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Validation report for dataset or annotations"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_checked: int = 0


# Training Models
@dataclass
class TrainingConfig:
    """Configuration for YOLO model training"""
    # Model parameters
    model_architecture: str  # e.g., 'yolov5s', 'yolov8n'
    pretrained_weights: Optional[Path] = None
    num_classes: int = 1
    class_names: List[str] = field(default_factory=list)
    task_type: str = 'detect'  # 'detect' or 'segment'
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    optimizer: str = 'SGD'
    
    # Data parameters
    dataset_yaml: Optional[Path] = None
    augmentation: bool = True
    
    # Hardware parameters
    device: str = 'cpu'  # 'cpu', 'cuda', 'mps'
    workers: int = 4
    
    # Checkpoint parameters
    save_period: int = 10
    resume_from: Optional[Path] = None


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch"""
    epoch: int
    train_loss: float
    val_loss: float
    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass
class TrainingResult:
    """Result of model training"""
    final_metrics: Dict[str, float]
    best_epoch: int
    training_time_seconds: float
    model_path: Path
    checkpoint_dir: Path
    training_history: List[EpochMetrics]
    is_simulation: bool = False  # Flag to indicate simulation mode
    archive_path: Optional[Path] = None  # Path to archived model with timestamp


# Evaluation Models
@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    visualize_results: bool = True
    save_predictions: bool = True


@dataclass
class Detection:
    """Single object detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass
class Metrics:
    """Evaluation metrics"""
    precision: float
    recall: float
    f1_score: float
    map50: float
    map50_95: float


@dataclass
class EvaluationResult:
    """Result of model evaluation"""
    overall_metrics: Metrics
    per_class_metrics: Dict[str, Metrics]
    inference_time_ms: float
    total_images: int
    predictions: List[List[Detection]]


# Licensing Models
@dataclass
class LicenseInfo:
    """License information"""
    license_type: str  # e.g., 'CC-BY-4.0', 'CC0', 'Custom'
    license_url: str
    attribution_required: bool
    commercial_use_allowed: bool
    modification_allowed: bool


@dataclass
class ImageLicense:
    """License information for a specific image"""
    image_id: str
    source_url: str
    license: LicenseInfo
    attribution_text: str
    collected_at: datetime


# Pipeline Models
class PipelineStage(Enum):
    """Pipeline execution stages"""
    DATA_COLLECTION = "data_collection"
    ANNOTATION_PROCESSING = "annotation_processing"
    DATASET_SPLITTING = "dataset_splitting"
    TRAINING = "training"
    EVALUATION = "evaluation"


@dataclass
class PipelineConfig:
    """Configuration for pipeline orchestration"""
    dataset_config: DatasetConfig
    training_config: TrainingConfig
    evaluation_config: EvaluationConfig
    stages_to_run: List[PipelineStage]


@dataclass
class StageResult:
    """Result of a single pipeline stage"""
    stage: PipelineStage
    success: bool
    duration_seconds: float
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    stages_completed: List[PipelineStage]
    stages_failed: List[PipelineStage]
    total_time_seconds: float
    final_model_path: Optional[Path] = None
    evaluation_results: Optional[EvaluationResult] = None
    errors: List[str] = field(default_factory=list)
