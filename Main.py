#!/usr/bin/env python3
"""
Main training/evaluation orchestrator for ABIDE SINDy ML pipeline.

Loads configuration, creates dataloaders, trains model, and evaluates.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from Helper import Config, print_config
from Data_processing import create_dataloaders
from Network import create_model, count_parameters


def setup_logging(log_dir: Path):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    
    return logging.getLogger(__name__)


class ModelTrainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Config,
        logger: logging.Logger,
        writer: SummaryWriter,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on (cpu or cuda)
            config: Configuration object
            logger: Logger instance
            writer: TensorBoard writer
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.logger = logger
        self.writer = writer
        
        # Training parameters
        self.lr = config.get("training.learning_rate", 0.001)
        self.weight_decay = config.get("training.weight_decay", 1e-5)
        self.num_epochs = config.get("training.num_epochs", 50)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=True,
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.global_step = 0
        
        self.logger.info(f"Model initialized with {count_parameters(model)} parameters")
        self.logger.info(f"Using device: {device}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training dataloader
        
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            node_feat = batch["node_features"].to(self.device)
            edge_feat = batch["edge_features"].to(self.device)
            tri_feat = batch["triangle_features"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(node_feat, edge_feat, tri_feat)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
            
            # Log to tensorboard
            self.writer.add_scalar("Loss/train", loss.item(), self.global_step)
            self.global_step += 1
        
        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
        
        self.logger.info(f"Train - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        
        return metrics
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation dataloader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            
            for batch in pbar:
                # Move to device
                node_feat = batch["node_features"].to(self.device)
                edge_feat = batch["edge_features"].to(self.device)
                tri_feat = batch["triangle_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                logits = self.model(node_feat, edge_feat, tri_feat)
                loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        
        self.logger.info(
            f"Validation - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        
        # Track best model
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
            self.best_model_state = self.model.state_dict().copy()
            self.logger.info(f"New best validation accuracy: {accuracy:.4f}")
        
        return metrics
    
    def test(self, test_loader) -> Dict[str, Any]:
        """
        Evaluate on test set.
        
        Args:
            test_loader: Test dataloader
        
        Returns:
            Dictionary with test metrics
        """
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Loaded best model for testing")
        
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            
            for batch in pbar:
                # Move to device
                node_feat = batch["node_features"].to(self.device)
                edge_feat = batch["edge_features"].to(self.device)
                tri_feat = batch["triangle_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                logits = self.model(node_feat, edge_feat, tri_feat)
                loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        
        # Compute metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
        }
        
        self.logger.info(
            f"Test - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        self.logger.info(f"Confusion Matrix:\n{cm}")
        
        return metrics
    
    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        
        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }
        
        self.logger.info(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            self.logger.info(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            
            # Validate
            val_metrics = self.validate(val_loader)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])
            
            # Log to tensorboard
            self.writer.add_scalars(
                "Loss",
                {"train": train_metrics["loss"], "val": val_metrics["loss"]},
                epoch
            )
            self.writer.add_scalars(
                "Accuracy",
                {"train": train_metrics["accuracy"], "val": val_metrics["accuracy"]},
                epoch
            )
            
            # Update learning rate
            self.scheduler.step(val_metrics["accuracy"])
        
        self.logger.info("\nTraining completed!")
        
        return history


def main():
    """Main entry point."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train and evaluate ABIDE SINDy ML model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (YAML or JSON)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SCCN_LSTM",
        choices=["SCCN_LSTM", "SCCN_Pool", "SCCN_Attention"],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Number of subjects to use (None for all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = Config.from_file(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = Config()
        print("Using default configuration")
    
    # Override config with command-line arguments
    if args.model:
        config.config["model"]["name"] = args.model
    if args.n_subjects:
        config.config["dataset"]["n_subjects"] = args.n_subjects
    if args.batch_size:
        config.config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config.config["training"]["num_epochs"] = args.epochs
    if args.lr:
        config.config["training"]["learning_rate"] = args.lr
    
    config.config["training"]["random_seed"] = args.seed
    
    # Print config
    print("\n" + "="*60)
    print_config(config)
    print("="*60 + "\n")
    
    # Setup logging
    log_dir = Path(config.get("logging.log_dir", "./logs"))
    logger = setup_logging(log_dir)
    
    # Setup TensorBoard
    tb_dir = log_dir / f"runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(str(tb_dir))
    
    logger.info("="*60)
    logger.info("ABIDE SINDy ML Pipeline")
    logger.info("="*60)
    
    # Setup device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        # Create dataloaders
        logger.info("Creating dataloaders...")
        upsample_factor = config.get("processing.upsample_factor", 100)
        window_length = int(
            config.get("processing.window_length_base", 10) *
            upsample_factor
        )
        
        train_loader, val_loader, test_loader, metadata = create_dataloaders(
            data_dir="/hdfs1/Data/Shubhajit/Project/New_TETON",
            n_subjects=config.get("dataset.n_subjects"),
            window_length=window_length,
            window_overlap=config.get("processing.window_overlap", 0.5),
            batch_size=config.get("training.batch_size", 32),
            train_split=config.get("training.train_split", 0.7),
            val_split=config.get("training.val_split", 0.15),
            test_split=config.get("training.test_split", 0.15),
            random_seed=args.seed,
            verbose=config.get("logging.verbose", True),
            upsample_factor=upsample_factor,
        )
        
        logger.info(f"Metadata: {metadata}")
        
        # Create model
        logger.info(f"Creating model: {config.get('model.name')}")
        model = create_model(
            model_name=config.get("model.name", "SCCN_LSTM"),
            input_dim=1,
            sccn_hidden=config.get("model.sccn_hidden", 64),
            num_sccn_layers=config.get("model.sccn_layers", 2),
            lstm_hidden=config.get("model.lstm_hidden", 128),
            lstm_layers=config.get("model.lstm_layers", 1),
            num_classes=config.get("model.output_classes", 2),
            dropout=config.get("model.dropout", 0.3),
        )
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            device=device,
            config=config,
            logger=logger,
            writer=writer,
        )
        
        # Train
        logger.info("Starting training...")
        history = trainer.train(train_loader, val_loader)
        
        # Test
        logger.info("Starting testing...")
        test_results = trainer.test(test_loader)
        
        # Save results
        checkpoint_dir = Path(config.get("logging.checkpoint_dir", "./checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = checkpoint_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results = {
            "config": config.to_dict(),
            "metadata": metadata,
            "history": history,
            "test_results": {k: v for k, v in test_results.items() if k != "probabilities"},
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save model
        model_file = checkpoint_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(trainer.best_model_state, model_file)
        logger.info(f"Model saved to {model_file}")
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise
    
    finally:
        writer.close()


if __name__ == "__main__":
    main()
