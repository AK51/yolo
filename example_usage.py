"""Example usage of YOLO training pipeline"""
from pathlib import Path

from src.annotation import AnnotationProcessor
from src.config import ConfigurationManager
from src.dataset import DatasetManager
from src.evaluation import EvaluationModule
from src.models import DatasetConfig, EvaluationConfig
from src.training import TrainingEngine


def main():
    """Example workflow"""
    
    # 1. Initialize dataset manager
    print("=" * 60)
    print("Step 1: Initialize Dataset Manager")
    print("=" * 60)
    
    dataset_root = Path("./data/my_dataset")
    config = DatasetConfig(
        min_image_width=100,
        min_image_height=100,
        check_duplicates=True
    )
    
    manager = DatasetManager(dataset_root, config)
    print(f"Dataset initialized at: {dataset_root}")
    
    # 2. Import local images (example)
    print("\n" + "=" * 60)
    print("Step 2: Import Local Images")
    print("=" * 60)
    print("To import images, use:")
    print("  result = manager.import_local_images(Path('/path/to/images'))")
    print("  Or use CLI: python -m src.cli collect --dataset-root ./data/my_dataset --source local --source-dir /path/to/images")
    
    # 3. Split dataset
    print("\n" + "=" * 60)
    print("Step 3: Split Dataset")
    print("=" * 60)
    print("To split dataset, use:")
    print("  result = manager.split_dataset(0.7, 0.2, 0.1, seed=42)")
    print("  Or use CLI: python -m src.cli split --dataset-root ./data/my_dataset --train 0.7 --val 0.2 --test 0.1")
    
    # 4. Configure training
    print("\n" + "=" * 60)
    print("Step 4: Configure Training")
    print("=" * 60)
    
    config_manager = ConfigurationManager("yolov5")
    training_config = config_manager.create_default_config("general")
    training_config.num_classes = 1
    training_config.class_names = ["object"]
    training_config.epochs = 50
    
    print(f"Training configuration:")
    print(f"  Model: {training_config.model_architecture}")
    print(f"  Epochs: {training_config.epochs}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Image size: {training_config.image_size}")
    
    # 5. Train model
    print("\n" + "=" * 60)
    print("Step 5: Train Model")
    print("=" * 60)
    print("To train model, use:")
    print("  engine = TrainingEngine(training_config, Path('./runs/train'))")
    print("  result = engine.train()")
    print("  Or use CLI: python -m src.cli train --config configs/yolov5_default.yaml --output ./runs/train")
    
    # 6. Evaluate model
    print("\n" + "=" * 60)
    print("Step 6: Evaluate Model")
    print("=" * 60)
    print("To evaluate model, use:")
    print("  eval_config = EvaluationConfig(confidence_threshold=0.25)")
    print("  evaluator = EvaluationModule(Path('./runs/train/best.pt'), eval_config)")
    print("  result = evaluator.evaluate('test')")
    print("  Or use CLI: python -m src.cli evaluate --model ./runs/train/best.pt --split test")
    
    print("\n" + "=" * 60)
    print("Complete Workflow Summary")
    print("=" * 60)
    print("""
1. Collect/Import images:
   python -m src.cli collect --dataset-root ./data/my_dataset --source local --source-dir /path/to/images

2. View statistics:
   python -m src.cli stats --dataset-root ./data/my_dataset

3. Split dataset:
   python -m src.cli split --dataset-root ./data/my_dataset --train 0.7 --val 0.2 --test 0.1

4. Train model:
   python -m src.cli train --config configs/yolov5_default.yaml --output ./runs/train --epochs 50

5. Evaluate model:
   python -m src.cli evaluate --model ./runs/train/best.pt --split test --report ./report.md
    """)


if __name__ == "__main__":
    main()
