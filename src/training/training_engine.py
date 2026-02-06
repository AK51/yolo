"""Training Engine for YOLO models"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models import EpochMetrics, TrainingConfig, TrainingResult


class TrainingEngine:
    """Executes YOLO model training"""
    
    def __init__(self, config: TrainingConfig, output_dir: Path):
        """
        Initialize training engine with configuration.
        
        Args:
            config: Training configuration
            output_dir: Output directory for models and logs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.training_history = []
    
    def train(self, resume: bool = False) -> TrainingResult:
        """
        Execute model training with real YOLO (Ultralytics).
        
        Args:
            resume: Whether to resume from checkpoint
            
        Returns:
            TrainingResult with training statistics
        """
        from ultralytics import YOLO
        
        print(f"🚀 Training {self.config.model_architecture} for {self.config.epochs} epochs...")
        print(f"📐 Image size: {self.config.image_size}, Batch size: {self.config.batch_size}")
        print(f"💻 Device: {self.config.device}")
        
        start_time = time.time()
        
        try:
            # Initialize YOLO model
            model_name = self.config.model_architecture
            if not model_name.endswith('.pt'):
                model_name += '.pt'
            
            print(f"📦 Loading model: {model_name}")
            model = YOLO(model_name)
            
            # Prepare training arguments
            # For output_dir like "./model", we want:
            # - Ultralytics to save to: ./model/
            # - Which becomes: ./model/train/weights/
            # So we set project=output_dir and name="train" to organize outputs
            train_args = {
                'data': str(self.config.dataset_yaml) if hasattr(self.config, 'dataset_yaml') else None,
                'epochs': self.config.epochs,
                'batch': self.config.batch_size,
                'imgsz': self.config.image_size,
                'device': self.config.device,
                'project': str(self.output_dir),  # Use full output_dir as project
                'name': 'train',  # Simple name to avoid confusion
                'exist_ok': True,
                'save_period': self.config.save_period if hasattr(self.config, 'save_period') else 10,
                'patience': self.config.patience if hasattr(self.config, 'patience') else 10,
                'workers': 0,  # CRITICAL FIX: Set to 0 to avoid Windows multiprocessing error
                'lr0': self.config.learning_rate if hasattr(self.config, 'learning_rate') else 0.01,
                'verbose': True,
            }
            
            # Remove None values
            train_args = {k: v for k, v in train_args.items() if v is not None}
            
            print(f"🎯 Starting training with Ultralytics YOLO...")
            print(f"📊 Configuration: {train_args}")
            
            # Train the model
            results = model.train(**train_args)
            
            training_time = time.time() - start_time
            
            # Extract metrics from results
            # Ultralytics stores results in the model's trainer
            best_map = 0.0
            best_epoch = 0
            
            if hasattr(results, 'results_dict'):
                metrics_dict = results.results_dict
                best_map = metrics_dict.get('metrics/mAP50(B)', 0.0)
            
            # Try to get metrics from saved results
            results_file = self.output_dir / 'results.csv'
            if results_file.exists():
                import csv
                with open(results_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        # Parse metrics from last row
                        for epoch_idx, row in enumerate(rows):
                            epoch_metrics = EpochMetrics(
                                epoch=epoch_idx + 1,
                                train_loss=float(row.get('train/box_loss', 0)),
                                val_loss=float(row.get('val/box_loss', 0)),
                                precision=float(row.get('metrics/precision(B)', 0)),
                                recall=float(row.get('metrics/recall(B)', 0)),
                                map50=float(row.get('metrics/mAP50(B)', 0)),
                                map50_95=float(row.get('metrics/mAP50-95(B)', 0))
                            )
                            self.training_history.append(epoch_metrics)
                            
                            if epoch_metrics.map50 > best_map:
                                best_map = epoch_metrics.map50
                                best_epoch = epoch_idx + 1
            
            # Find the best model file
            # With project=output_dir and name="train", Ultralytics creates:
            # output_dir/train/weights/best.pt (no "detect" folder when project is absolute path)
            # But we also need to check for runs/segment/model/train/weights/best.pt
            possible_paths = [
                Path("./runs/segment/model/train/weights/best.pt"),  # Relative path for segmentation
                Path("./runs/detect/train/weights/best.pt"),  # Relative path for detection
                self.output_dir / "train" / "weights" / "best.pt",  # New structure (absolute path)
                self.output_dir / "detect" / "train" / "weights" / "best.pt",  # With detect folder
                self.output_dir / "weights" / "best.pt",  # Direct path
                self.output_dir.parent / "detect" / self.output_dir.parent.name / self.output_dir.name / "weights" / "best.pt",  # Old structure
            ]
            
            model_path = None
            print(f"\n🔍 Searching for trained model...")
            for path in possible_paths:
                print(f"   Checking: {path}")
                if path.exists():
                    file_size = path.stat().st_size / (1024 * 1024)
                    print(f"   ✅ Found! Size: {file_size:.2f} MB")
                    # Only accept files larger than 1 MB (real models are 50+ MB)
                    if file_size > 1.0:
                        model_path = path
                        print(f"   ✅ Valid model file (size > 1 MB)")
                        break
                    else:
                        print(f"   ⚠️  File too small ({file_size:.2f} MB), skipping...")
                else:
                    print(f"   ❌ Not found")
            
            if model_path is None:
                print("\n⚠️ Warning: Could not find best.pt, trying last.pt")
                for path in possible_paths:
                    last_path = path.parent / "last.pt"
                    if last_path.exists():
                        file_size = last_path.stat().st_size / (1024 * 1024)
                        if file_size > 1.0:
                            model_path = last_path
                            print(f"   ✅ Found last.pt: {last_path} ({file_size:.2f} MB)")
                            break
            
            if model_path is None:
                print("\n❌ Error: Could not find trained model files")
                print(f"   Searched in:")
                for path in possible_paths:
                    print(f"     - {path}")
                print(f"\n⚠️  This usually means training failed or model wasn't saved correctly")
                # Don't use placeholder - return error instead
                raise FileNotFoundError("Trained model file not found")
            
            print(f"\n✅ Using model: {model_path}")
            print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save training history
            self._save_training_history()
            
            # Get final metrics
            final_metrics = {
                'map50': best_map,
                'map50_95': self.training_history[-1].map50_95 if self.training_history else 0.0,
                'precision': self.training_history[-1].precision if self.training_history else 0.0,
                'recall': self.training_history[-1].recall if self.training_history else 0.0
            }
            
            result = TrainingResult(
                final_metrics=final_metrics,
                best_epoch=best_epoch if best_epoch > 0 else self.config.epochs,
                training_time_seconds=training_time,
                model_path=model_path,
                checkpoint_dir=self.checkpoint_dir,
                training_history=self.training_history,
                is_simulation=False  # Real training
            )
            
            print(f"\n✅ Training complete! Best mAP50: {best_map:.4f} at epoch {best_epoch}")
            if model_path and model_path.exists():
                print(f"💾 Model saved to: {model_path}")
                print(f"📊 Model size: {model_path.stat().st_size / (1024*1024):.2f} MB")
                
                # Copy best model to output directory with timestamp
                try:
                    import shutil
                    from datetime import datetime
                    import torch
                    
                    print("\n" + "=" * 70)
                    print("🔍 VERIFYING TRAINED MODEL")
                    print("=" * 70)
                    
                    # Create model directory if it doesn't exist
                    model_archive_dir = Path("./model")
                    model_archive_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Check if this is a segmentation model
                    is_segmentation = False
                    print(f"📂 Loading model checkpoint from: {model_path}")
                    print(f"📊 File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
                    
                    try:
                        checkpoint = torch.load(str(model_path), map_location='cpu')
                        print(f"✅ Model checkpoint loaded successfully")
                        
                        # Get training arguments
                        train_args = checkpoint.get('train_args', {})
                        actual_task = train_args.get('task', 'unknown')
                        actual_model = train_args.get('model', 'unknown')
                        
                        print(f"\n📋 Model Training Arguments:")
                        print(f"   Task: {actual_task}")
                        print(f"   Model: {actual_model}")
                        print(f"   Epochs: {train_args.get('epochs', 'unknown')}")
                        print(f"   Batch: {train_args.get('batch', 'unknown')}")
                        print(f"   Image size: {train_args.get('imgsz', 'unknown')}")
                        print(f"   Device: {train_args.get('device', 'unknown')}")
                        print(f"   Workers: {train_args.get('workers', 'unknown')}")
                        
                        # Get model info
                        if 'model' in checkpoint:
                            model_state = checkpoint['model']
                            if hasattr(model_state, 'names'):
                                print(f"\n📝 Class Names: {model_state.names}")
                        
                        # Get training metrics
                        if 'train_metrics' in checkpoint:
                            metrics = checkpoint['train_metrics']
                            print(f"\n📊 Training Metrics:")
                            if 'metrics/mAP50(B)' in metrics:
                                print(f"   Box mAP50: {metrics['metrics/mAP50(B)']:.4f}")
                            if 'metrics/mAP50-95(B)' in metrics:
                                print(f"   Box mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
                            if 'metrics/mAP50(M)' in metrics:
                                print(f"   Mask mAP50: {metrics['metrics/mAP50(M)']:.4f}")
                            if 'metrics/mAP50-95(M)' in metrics:
                                print(f"   Mask mAP50-95: {metrics['metrics/mAP50-95(M)']:.4f}")
                        
                        # Verify if it's a segmentation model
                        print(f"\n🎯 Model Type Verification:")
                        print(f"   Checking task type: {actual_task}")
                        print(f"   Checking model name: {actual_model}")
                        
                        if actual_task == 'segment':
                            print(f"   ✅ Task is 'segment' - This is a segmentation model!")
                        else:
                            print(f"   ❌ Task is '{actual_task}' - This is NOT a segmentation model!")
                        
                        if '-seg' in str(actual_model):
                            print(f"   ✅ Model has '-seg' suffix - Correct architecture!")
                        else:
                            print(f"   ❌ Model doesn't have '-seg' suffix - Wrong architecture!")
                        
                        if actual_task == 'segment' and '-seg' in str(actual_model):
                            is_segmentation = True
                            print(f"\n✅✅✅ CONFIRMED: This IS a SEGMENTATION model! ✅✅✅")
                        else:
                            print(f"\n❌ This is NOT a segmentation model")
                            if actual_task == 'detect':
                                print(f"   This is a DETECTION model")
                        
                    except Exception as e:
                        print(f"❌ Error verifying model type: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Generate timestamped filename with _seg suffix if segmentation
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if is_segmentation:
                        model_name = f"best_seg_{timestamp}.pt"
                        print(f"\n📝 Filename: {model_name} (with '_seg' suffix)")
                    else:
                        model_name = f"best_{timestamp}.pt"
                        print(f"\n📝 Filename: {model_name} (no '_seg' suffix)")
                    
                    archive_path = model_archive_dir / model_name
                    
                    # Copy the model
                    print(f"\n📦 Copying model to archive...")
                    print(f"   Source: {model_path}")
                    print(f"   Destination: {archive_path}")
                    
                    shutil.copy2(model_path, archive_path)
                    
                    # Verify the copy
                    if archive_path.exists():
                        archive_size = archive_path.stat().st_size / (1024*1024)
                        print(f"✅ Model copied successfully!")
                        print(f"   Archive size: {archive_size:.2f} MB")
                        
                        # Double-check the archived model
                        try:
                            verify_checkpoint = torch.load(str(archive_path), map_location='cpu')
                            verify_task = verify_checkpoint.get('train_args', {}).get('task', 'unknown')
                            print(f"✅ Archived model verified: task={verify_task}")
                        except Exception as e:
                            print(f"⚠️ Could not verify archived model: {e}")
                    else:
                        print(f"❌ Error: Archived model not found!")
                    
                    print("=" * 70)
                    print(f"📦 Model archived to: {archive_path}")
                    print("=" * 70 + "\n")
                    
                    # Update result to include archive path
                    result.archive_path = archive_path
                    
                except Exception as e:
                    print(f"❌ Error during model archiving: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ Warning: Model file not found at expected location")
                if model_path:
                    print(f"   Expected: {model_path}")
                print(f"   Ultralytics should have saved to: {self.output_dir / 'train' / 'weights' / 'best.pt'}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            print(f"⚠️ Falling back to simulation mode...")
            import traceback
            traceback.print_exc()
            return self._train_simulation(resume)
    
    def _train_simulation(self, resume: bool = False) -> TrainingResult:
        """
        Fallback simulation mode for testing without real training.
        
        Args:
            resume: Whether to resume from checkpoint
            
        Returns:
            TrainingResult with simulated statistics
        """
        print("\n" + "=" * 60)
        print("⚠️  SIMULATION MODE ACTIVE")
        print("=" * 60)
        print("PyTorch/GPU is not available due to DLL errors.")
        print("This is a known Windows compatibility issue with PyTorch 2.10.0")
        print("\nSimulation will:")
        print("  ✓ Show training progress")
        print("  ✓ Generate metrics and logs")
        print("  ✓ Create checkpoint files")
        print("  ✗ NOT train a real model")
        print("\nTo fix and enable real GPU training:")
        print("  1. Close the GUI")
        print("  2. Run: .\\fix_dll_error_final.bat")
        print("  3. Restart the GUI")
        print("=" * 60)
        print()
        
        print(f"⚠️ SIMULATION: Training {self.config.model_architecture} for {self.config.epochs} epochs...")
        print(f"📐 Image size: {self.config.image_size}, Batch size: {self.config.batch_size}")
        print(f"💻 Device: {self.config.device} (simulated)")
        
        start_time = time.time()
        start_epoch = 0
        
        if resume and self.config.resume_from:
            start_epoch = self._load_checkpoint(self.config.resume_from)
            print(f"Resuming from epoch {start_epoch}")
        
        best_map = 0.0
        best_epoch = 0
        
        for epoch in range(start_epoch, self.config.epochs):
            # Simulate training with realistic metrics
            epoch_metrics = EpochMetrics(
                epoch=epoch + 1,
                train_loss=0.5 - (epoch * 0.001),  # Simulated decreasing loss
                val_loss=0.6 - (epoch * 0.001),
                precision=0.7 + (epoch * 0.002),
                recall=0.65 + (epoch * 0.002),
                map50=0.6 + (epoch * 0.003),
                map50_95=0.4 + (epoch * 0.002)
            )
            
            self.training_history.append(epoch_metrics)
            
            # Track best model
            if epoch_metrics.map50 > best_map:
                best_map = epoch_metrics.map50
                best_epoch = epoch + 1
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.save_period == 0:
                self._save_checkpoint(epoch + 1, epoch_metrics)
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[SIMULATION] Epoch {epoch + 1}/{self.config.epochs} - "
                      f"Loss: {epoch_metrics.train_loss:.4f}, "
                      f"mAP50: {epoch_metrics.map50:.4f}")
        
        training_time = time.time() - start_time
        
        # Save final model (empty file in simulation)
        model_path = self.output_dir / "best.pt"
        model_path.touch()
        
        # Save training history
        self._save_training_history()
        
        result = TrainingResult(
            final_metrics={
                'map50': best_map,
                'map50_95': self.training_history[-1].map50_95,
                'precision': self.training_history[-1].precision,
                'recall': self.training_history[-1].recall
            },
            best_epoch=best_epoch,
            training_time_seconds=training_time,
            model_path=model_path,
            checkpoint_dir=self.checkpoint_dir,
            training_history=self.training_history,
            is_simulation=True,  # Simulation mode
            archive_path=None  # No archiving for simulation
        )
        
        print(f"\n⚠️ SIMULATION complete! Best mAP50: {best_map:.4f} at epoch {best_epoch}")
        print(f"💾 Simulation files saved to: {self.output_dir}")
        print(f"\n⚠️ REMINDER: This was a simulation. No real model was trained.")
        print(f"   Run .\\fix_dll_error_final.bat to enable real GPU training.")
        
        return result
    
    def _save_checkpoint(self, epoch: int, metrics: EpochMetrics) -> None:
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
        checkpoint_data = {
            'epoch': epoch,
            'metrics': vars(metrics),
            'config': vars(self.config)
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
    
    def _load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load checkpoint and return starting epoch"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        return checkpoint_data.get('epoch', 0)
    
    def _save_training_history(self) -> None:
        """Save training history to JSON"""
        history_path = self.output_dir / "training_history.json"
        history_data = [vars(m) for m in self.training_history]
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
