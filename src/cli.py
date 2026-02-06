"""Command-line interface for YOLO training pipeline"""
import argparse
import sys
from pathlib import Path

from src.annotation import AnnotationProcessor
from src.config import ConfigurationManager
from src.dataset import DatasetManager
from src.evaluation import EvaluationModule
from src.models import DatasetConfig, EvaluationConfig
from src.training import TrainingEngine


def cmd_collect(args):
    """Handle collect command"""
    print(f"Collecting dataset from {args.source}...")
    
    config = DatasetConfig()
    manager = DatasetManager(args.dataset_root, config)
    
    if args.source == "local":
        if not args.source_dir:
            print("Error: --source-dir required for local collection")
            return 1
        
        result = manager.import_local_images(Path(args.source_dir), copy=True)
        print(f"Collected {result.images_collected} images")
        if result.images_failed > 0:
            print(f"Failed: {result.images_failed} images")
        if result.errors:
            print("Errors:")
            for error in result.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
    
    elif args.source == "coco":
        categories = args.categories.split(',') if args.categories else []
        result = manager.collect_from_coco(categories, args.max_images)
        print(f"Collected {result.images_collected} images from COCO")
    
    elif args.source == "open-images":
        categories = args.categories.split(',') if args.categories else []
        result = manager.collect_from_open_images(categories, args.max_images)
        print(f"Collected {result.images_collected} images from Open Images")
    
    return 0


def cmd_split(args):
    """Handle split command"""
    print(f"Splitting dataset: train={args.train}, val={args.val}, test={args.test}")
    
    config = DatasetConfig()
    manager = DatasetManager(args.dataset_root, config)
    
    result = manager.split_dataset(
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
    
    print(f"Split complete:")
    print(f"  Train: {result['train']} images")
    print(f"  Val: {result['val']} images")
    print(f"  Test: {result['test']} images")
    
    return 0


def cmd_annotate(args):
    """Handle annotate command"""
    print(f"Converting annotations from {args.format} to YOLO...")
    
    # Load class mapping
    class_mapping = {0: "object"}  # Default
    if args.class_mapping:
        import json
        with open(args.class_mapping, 'r') as f:
            class_mapping = json.load(f)
    
    processor = AnnotationProcessor(class_mapping)
    
    if args.format == "coco":
        result = processor.convert_coco_to_yolo(
            Path(args.input),
            Path(args.images_dir),
            Path(args.output)
        )
    elif args.format == "voc":
        result = processor.convert_voc_to_yolo(
            Path(args.input),
            Path(args.images_dir),
            Path(args.output)
        )
    else:
        print(f"Unsupported format: {args.format}")
        return 1
    
    print(f"Converted {result.annotations_converted} annotations")
    print(f"Processed {result.images_processed} images")
    if result.annotations_failed > 0:
        print(f"Failed: {result.annotations_failed} annotations")
    
    return 0


def cmd_train(args):
    """Handle train command"""
    print(f"Training YOLO model...")
    
    # Load configuration
    config_manager = ConfigurationManager(args.yolo_version)
    
    if args.config:
        config = config_manager.load_config(Path(args.config))
    else:
        config = config_manager.create_default_config("general")
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.image_size = args.img_size
        config.device = args.device
    
    # Create training engine
    engine = TrainingEngine(config, Path(args.output))
    
    # Train
    result = engine.train(resume=args.resume)
    
    print(f"\nTraining Results:")
    print(f"  Best mAP50: {result.final_metrics['map50']:.4f}")
    print(f"  Training time: {result.training_time_seconds:.1f}s")
    print(f"  Model saved: {result.model_path}")
    
    return 0


def cmd_evaluate(args):
    """Handle evaluate command"""
    print(f"Evaluating model...")
    
    config = EvaluationConfig(
        confidence_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    evaluator = EvaluationModule(Path(args.model), config)
    result = evaluator.evaluate(args.split)
    
    # Generate report
    if args.report:
        evaluator.generate_report(result, Path(args.report))
    
    return 0


def cmd_stats(args):
    """Handle stats command"""
    print(f"Dataset statistics for: {args.dataset_root}")
    
    config = DatasetConfig()
    manager = DatasetManager(args.dataset_root, config)
    
    stats = manager.get_statistics()
    
    print(f"\nDataset Statistics:")
    print(f"  Total images: {stats.total_images}")
    print(f"  Total size: {stats.total_size_bytes / (1024*1024):.2f} MB")
    print(f"  Train: {stats.train_count}")
    print(f"  Val: {stats.val_count}")
    print(f"  Test: {stats.test_count}")
    
    if stats.class_distribution:
        print(f"\nClass Distribution:")
        for class_name, count in stats.class_distribution.items():
            print(f"  {class_name}: {count}")
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="YOLO Object Detection Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect dataset')
    collect_parser.add_argument('--dataset-root', required=True, help='Dataset root directory')
    collect_parser.add_argument('--source', required=True, choices=['local', 'coco', 'open-images'],
                               help='Data source')
    collect_parser.add_argument('--source-dir', help='Source directory (for local)')
    collect_parser.add_argument('--categories', help='Comma-separated categories')
    collect_parser.add_argument('--max-images', type=int, default=300, help='Maximum images to collect')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset')
    split_parser.add_argument('--dataset-root', required=True, help='Dataset root directory')
    split_parser.add_argument('--train', type=float, default=0.7, help='Train ratio')
    split_parser.add_argument('--val', type=float, default=0.2, help='Validation ratio')
    split_parser.add_argument('--test', type=float, default=0.1, help='Test ratio')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Annotate command
    annotate_parser = subparsers.add_parser('annotate', help='Convert annotations')
    annotate_parser.add_argument('--format', required=True, choices=['coco', 'voc'],
                                help='Input annotation format')
    annotate_parser.add_argument('--input', required=True, help='Input annotations path')
    annotate_parser.add_argument('--images-dir', required=True, help='Images directory')
    annotate_parser.add_argument('--output', required=True, help='Output directory')
    annotate_parser.add_argument('--class-mapping', help='Class mapping JSON file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train YOLO model')
    train_parser.add_argument('--config', help='Training configuration file')
    train_parser.add_argument('--yolo-version', default='yolov5', choices=['yolov5', 'yolov8'],
                             help='YOLO version')
    train_parser.add_argument('--output', default='./runs/train', help='Output directory')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--img-size', type=int, default=640, help='Image size')
    train_parser.add_argument('--device', default='cpu', help='Device (cpu, cuda)')
    train_parser.add_argument('--resume', action='store_true', help='Resume training')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', required=True, help='Model path')
    eval_parser.add_argument('--split', default='test', choices=['val', 'test'],
                            help='Dataset split')
    eval_parser.add_argument('--conf-threshold', type=float, default=0.25,
                            help='Confidence threshold')
    eval_parser.add_argument('--iou-threshold', type=float, default=0.45,
                            help='IOU threshold')
    eval_parser.add_argument('--report', help='Output report path')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.add_argument('--dataset-root', required=True, help='Dataset root directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    commands = {
        'collect': cmd_collect,
        'split': cmd_split,
        'annotate': cmd_annotate,
        'train': cmd_train,
        'evaluate': cmd_evaluate,
        'stats': cmd_stats
    }
    
    try:
        return commands[args.command](args)
    except Exception as e:
        print(f"Error: {str(e)}")
        if '--verbose' in sys.argv:
            raise
        return 1


if __name__ == '__main__':
    sys.exit(main())
