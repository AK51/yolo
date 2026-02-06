"""Pipeline Orchestrator for YOLO Training Pipeline"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.annotation.annotation_processor import AnnotationProcessor
from src.config.config_manager import ConfigurationManager
from src.dataset.dataset_manager import DatasetManager
from src.evaluation.evaluation_module import EvaluationModule
from src.logging_config import get_pipeline_logger
from src.models import (
    EvaluationResult,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    TrainingConfig,
    ValidationReport,
)
from src.training.training_engine import TrainingEngine


class PipelineOrchestrator:
    """Orchestrates the complete YOLO training pipeline"""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = get_pipeline_logger(verbose=config.verbose if hasattr(config, 'verbose') else False)
        self.component_logger = self.logger.get_logger('pipeline_orchestrator')
        
        # Initialize components
        self.dataset_manager: Optional[DatasetManager] = None
        self.annotation_processor: Optional[AnnotationProcessor] = None
        self.config_manager: Optional[ConfigurationManager] = None
        self.training_engine: Optional[TrainingEngine] = None
        self.evaluation_module: Optional[EvaluationModule] = None
        
        # Track execution state
        self.start_time: Optional[datetime] = None
        self.stages_completed: List[PipelineStage] = []
        self.stages_failed: List[PipelineStage] = []
        self.errors: List[str] = []
    
    def run_full_pipeline(self) -> PipelineResult:
        """
        Execute complete pipeline from data collection to evaluation.
        
        Returns:
            PipelineResult with execution summary
        """
        self.component_logger.info("Starting full pipeline execution")
        self.start_time = datetime.now()
        
        # Execute all stages in sequence
        stages = self.config.stages_to_run if self.config.stages_to_run else [
            PipelineStage.DATA_COLLECTION,
            PipelineStage.ANNOTATION_PROCESSING,
            PipelineStage.DATASET_SPLITTING,
            PipelineStage.TRAINING,
            PipelineStage.EVALUATION
        ]
        
        for stage in stages:
            try:
                self.component_logger.info(f"Executing stage: {stage.value}")
                self._execute_stage(stage)
                self.stages_completed.append(stage)
                self.logger.log_success('pipeline_orchestrator', f"Stage {stage.value}")
            except Exception as e:
                self.component_logger.error(f"Stage {stage.value} failed: {str(e)}")
                self.stages_failed.append(stage)
                self.errors.append(f"{stage.value}: {str(e)}")
                self.logger.log_error('pipeline_orchestrator', f"Stage {stage.value}", e)
                break  # Halt on failure
        
        # Calculate total time
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        # Get final model path if training completed
        final_model_path = None
        if PipelineStage.TRAINING in self.stages_completed and self.training_engine:
            final_model_path = getattr(self.training_engine, 'model_path', None)
        
        # Get evaluation results if evaluation completed
        evaluation_results = None
        if PipelineStage.EVALUATION in self.stages_completed and self.evaluation_module:
            evaluation_results = getattr(self.evaluation_module, 'last_results', None)
        
        result = PipelineResult(
            stages_completed=[s.value for s in self.stages_completed],
            stages_failed=[s.value for s in self.stages_failed],
            total_time_seconds=total_time,
            final_model_path=final_model_path,
            evaluation_results=evaluation_results,
            errors=self.errors
        )
        
        self.component_logger.info(f"Pipeline execution completed in {total_time:.2f}s")
        return result
    
    def run_from_stage(self, starting_stage: PipelineStage) -> PipelineResult:
        """
        Execute pipeline starting from specified stage.
        
        Args:
            starting_stage: Stage to start execution from
            
        Returns:
            PipelineResult with execution summary
        """
        self.component_logger.info(f"Starting pipeline from stage: {starting_stage.value}")
        
        # Get all stages
        all_stages = [
            PipelineStage.DATA_COLLECTION,
            PipelineStage.ANNOTATION_PROCESSING,
            PipelineStage.DATASET_SPLITTING,
            PipelineStage.TRAINING,
            PipelineStage.EVALUATION
        ]
        
        # Find starting index
        try:
            start_idx = all_stages.index(starting_stage)
        except ValueError:
            raise ValueError(f"Invalid starting stage: {starting_stage}")
        
        # Execute from starting stage onwards
        stages_to_run = all_stages[start_idx:]
        
        # Temporarily override config
        original_stages = self.config.stages_to_run
        self.config.stages_to_run = stages_to_run
        
        result = self.run_full_pipeline()
        
        # Restore original config
        self.config.stages_to_run = original_stages
        
        return result
    
    def _validate_prerequisites(self, stage: PipelineStage) -> ValidationReport:
        """
        Validate that prerequisites are met for stage execution.
        
        Args:
            stage: Stage to validate prerequisites for
            
        Returns:
            ValidationReport with validation results
        """
        errors = []
        warnings = []
        
        if stage == PipelineStage.DATA_COLLECTION:
            # No prerequisites for data collection
            pass
        
        elif stage == PipelineStage.ANNOTATION_PROCESSING:
            # Check if dataset exists
            if not self.config.dataset_config.dataset_root.exists():
                errors.append("Dataset root directory does not exist")
        
        elif stage == PipelineStage.DATASET_SPLITTING:
            # Check if dataset has images
            if self.dataset_manager:
                if self.dataset_manager.manifest.total_images == 0:
                    errors.append("Dataset has no images to split")
        
        elif stage == PipelineStage.TRAINING:
            # Check if dataset is prepared
            if not self.config.training_config.dataset_yaml.exists():
                errors.append(f"Dataset YAML not found: {self.config.training_config.dataset_yaml}")
            
            # Check if dataset has train/val splits
            dataset_root = self.config.dataset_config.dataset_root
            train_dir = dataset_root / "images" / "train"
            val_dir = dataset_root / "images" / "val"
            
            if not train_dir.exists() or not list(train_dir.glob("*")):
                errors.append("Training images not found")
            if not val_dir.exists() or not list(val_dir.glob("*")):
                warnings.append("Validation images not found")
        
        elif stage == PipelineStage.EVALUATION:
            # Check if trained model exists
            if not hasattr(self, 'training_engine') or not self.training_engine:
                errors.append("Training engine not initialized - run training first")
        
        return ValidationReport(
            is_valid=len(errors) == 0,
            total_checked=1,
            errors=errors,
            warnings=warnings
        )
    
    def _execute_stage(self, stage: PipelineStage) -> None:
        """
        Execute a single pipeline stage.
        
        Args:
            stage: Stage to execute
        """
        # Validate prerequisites
        validation = self._validate_prerequisites(stage)
        if not validation.is_valid:
            raise RuntimeError(f"Prerequisites not met: {', '.join(validation.errors)}")
        
        # Log warnings
        for warning in validation.warnings:
            self.logger.log_warning('pipeline_orchestrator', warning)
        
        # Execute stage
        if stage == PipelineStage.DATA_COLLECTION:
            self._execute_data_collection()
        
        elif stage == PipelineStage.ANNOTATION_PROCESSING:
            self._execute_annotation_processing()
        
        elif stage == PipelineStage.DATASET_SPLITTING:
            self._execute_dataset_splitting()
        
        elif stage == PipelineStage.TRAINING:
            self._execute_training()
        
        elif stage == PipelineStage.EVALUATION:
            self._execute_evaluation()
        
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def _execute_data_collection(self) -> None:
        """Execute data collection stage"""
        self.component_logger.info("Executing data collection")
        
        # Initialize dataset manager
        self.dataset_manager = DatasetManager(
            self.config.dataset_config.dataset_root,
            self.config.dataset_config
        )
        
        # Collect from configured sources
        # This is a placeholder - actual implementation would collect from COCO, Open Images, etc.
        self.logger.log_info('pipeline_orchestrator', "Data collection stage completed")
    
    def _execute_annotation_processing(self) -> None:
        """Execute annotation processing stage"""
        self.component_logger.info("Executing annotation processing")
        
        # Initialize annotation processor if not already done
        if not self.annotation_processor:
            self.annotation_processor = AnnotationProcessor({})
        
        # Process annotations
        # This is a placeholder - actual implementation would convert annotations
        self.logger.log_info('pipeline_orchestrator', "Annotation processing stage completed")
    
    def _execute_dataset_splitting(self) -> None:
        """Execute dataset splitting stage"""
        self.component_logger.info("Executing dataset splitting")
        
        # Initialize dataset manager if not already done
        if not self.dataset_manager:
            self.dataset_manager = DatasetManager(
                self.config.dataset_config.dataset_root,
                self.config.dataset_config
            )
        
        # Split dataset
        split_config = getattr(self.config, 'split_ratios', (0.7, 0.2, 0.1))
        split_result = self.dataset_manager.split_dataset(
            train_ratio=split_config[0],
            val_ratio=split_config[1],
            test_ratio=split_config[2],
            seed=getattr(self.config, 'random_seed', 42)
        )
        
        self.logger.log_success(
            'pipeline_orchestrator',
            "Dataset splitting",
            f"Train: {split_result['train']}, Val: {split_result['val']}, Test: {split_result['test']}"
        )
    
    def _execute_training(self) -> None:
        """Execute training stage"""
        self.component_logger.info("Executing training")
        
        # Initialize training engine
        self.training_engine = TrainingEngine(
            self.config.training_config,
            self.config.training_config.output_dir
        )
        
        # Train model
        training_result = self.training_engine.train()
        
        self.logger.log_success(
            'pipeline_orchestrator',
            "Model training",
            f"Best mAP: {training_result.final_metrics.get('map50', 0):.4f}"
        )
    
    def _execute_evaluation(self) -> None:
        """Execute evaluation stage"""
        self.component_logger.info("Executing evaluation")
        
        # Get model path from training
        if not self.training_engine:
            raise RuntimeError("Training engine not initialized")
        
        model_path = getattr(self.training_engine, 'model_path', None)
        if not model_path or not Path(model_path).exists():
            raise RuntimeError("Trained model not found")
        
        # Initialize evaluation module
        from src.models import EvaluationConfig
        eval_config = EvaluationConfig(
            confidence_threshold=0.25,
            iou_threshold=0.45,
            max_detections=100,
            visualize_results=True,
            save_predictions=True
        )
        
        self.evaluation_module = EvaluationModule(model_path, eval_config)
        
        # Evaluate model
        eval_result = self.evaluation_module.evaluate(dataset_split='test')
        
        self.logger.log_success(
            'pipeline_orchestrator',
            "Model evaluation",
            f"mAP50: {eval_result.overall_metrics.map50:.4f}"
        )
