# Automatically fetch json logs, record new logs and log them to tracking systems

import json
import time
import os
from pathlib import Path
import shutil
import datetime
import sys
from typing import Dict, Any, List, Optional

import logging
logging.basicConfig(level=logging.INFO)

# Import logger classes
from src.exp_logging import BaseLogger, create_logger

from dotenv import load_dotenv
load_dotenv()

class LogFetcher:
    def __init__(self, 
                 log_file_path, 
                 logger: BaseLogger = None,
                 checkpoint_dir=None, 
                 last_line=0):
        self.log_file_path = Path(log_file_path)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.last_line = last_line
        self.last_checkpoint = None
        self.known_checkpoints = set()
        self.logger = logger
        
    def fetch_new_logs(self):
        """Fetch new logs from the training process."""
        if not self.log_file_path.exists():
            return []
        
        try:

            log_data = []

            # When the log file is empty, we can skip reading it
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r') as f:
                    for line in f:
                        log_data.append(json.loads(line))
                    
                logs = log_data
                new_logs = logs[self.last_line:]
                self.last_line = len(logs)
                return new_logs
            
            return []

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Error reading log file: {e}")
            return []
    
    def detect_new_checkpoints(self):
        """Detect if new checkpoint files have been created."""
        if not self.checkpoint_dir or not self.checkpoint_dir.exists():
            return None
            
        checkpoints = [
            cp for cp in self.checkpoint_dir.glob("checkpoint-*") 
            if cp.is_dir() and cp not in self.known_checkpoints
        ]
        
        if not checkpoints:
            return None
            
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        newest_checkpoint = checkpoints[0]
        
        # Check if this is actually a new checkpoint
        if newest_checkpoint != self.last_checkpoint:
            self.last_checkpoint = newest_checkpoint
            self.known_checkpoints.add(newest_checkpoint)
            return newest_checkpoint
        
        return None


def log_metrics(logger: BaseLogger, logs: List[Dict[str, Any]]):
    """
    Send logs to the configured tracking system (WandB or MLflow)
    
    Args:
        logger: BaseLogger instance (WandbLogger or MLflowLogger)
        logs: List of log dictionaries containing metrics
    """
    if not logger:
        logging.info(f"No logger configured, skipping metrics: {logs}")
        return
    
    for log_entry in logs:
        step = log_entry.get('step', 0)
        
        # Log metrics
        metrics = {}
        for key, value in log_entry.items():
            if isinstance(value, (int, float)) and key != 'step':
                metrics[key] = value
        
        if metrics:
            logger.log_metrics(metrics)
            logging.info(f"Logged metrics: {metrics} at step {step}")
            
        # Log any non-numeric values as separate artifacts if needed
        for key, value in log_entry.items():
            if not isinstance(value, (int, float)) and key not in ('step', 'timestamp'):
                # For complex objects, we could store them as JSON
                if isinstance(value, dict) or isinstance(value, list):
                    logger.log_metric(f"{key}_present", 1.0)


def upload_checkpoint(
    logger: BaseLogger, 
    checkpoint_path: Path,
    cloud_storage_path: str = None, 
    compress: bool = True,
    register_to_registry: bool = True,
    collection_name: str = None,
    registry_name: str = "model"
) -> Optional[str]:
    """
    Upload checkpoint to tracking system and optionally to cloud storage
    
    Args:
        logger: BaseLogger instance
        checkpoint_path: Path to the checkpoint directory
        cloud_storage_path: Optional cloud storage path
        compress: Whether to compress the checkpoint as a tarball (True) or upload raw (False)
        
    Returns:
        Name of the uploaded checkpoint or None if failed
    """
    if not logger:
        logging.warning(f"No logger configured, skipping checkpoint upload: {checkpoint_path}")
        return None
    
    checkpoint_name = checkpoint_path.name

    if not collection_name:
        collection_name = os.getenv("WANDB_REGISTRY", "default")
    
    try:
        if register_to_registry and logger.tracking_backend == "wandb":
            # Use WandB model registry feature
            logging.info(f"Registering checkpoint {checkpoint_path} to WandB registry...")
            registry_path = logger.register_model(
                model_path=str(checkpoint_path),
                model_name=checkpoint_name,
                collection_name=collection_name,
                registry_name=registry_name
            )
            logging.info(f"Model registered at: {registry_path}")
            return checkpoint_name
            
        elif compress:
            # Create a compressed tarball of the checkpoint
            tarball_path = f"/tmp/{checkpoint_name}.tar.gz"
            
            logging.info(f"Creating tarball of checkpoint {checkpoint_path}...")
            import tarfile
            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(checkpoint_path, arcname=checkpoint_name)
            
            # Log tarball as artifact
            logger.log_artifact(tarball_path, name=f"checkpoint-{checkpoint_name}")
            logging.info(f"Compressed checkpoint {checkpoint_name} uploaded to tracking system")
            
            # Clean up the temporary tarball
            os.remove(tarball_path)
        else:
            # Log the raw directory as an artifact
            logging.info(f"Uploading raw checkpoint directory {checkpoint_path}...")
            logger.log_directory(str(checkpoint_path), name=f"checkpoint-{checkpoint_name}")
            logging.info(f"Raw checkpoint {checkpoint_name} uploaded to tracking system")
        
        # Additional cloud storage upload if needed
        if cloud_storage_path:
            # Placeholder for cloud upload
            logging.info(f"Uploading checkpoint to cloud storage: {cloud_storage_path}")
            # This would typically use cloud SDK (AWS S3, GCP, etc.)
        
        return checkpoint_name
    
    except Exception as e:
        logging(f"Error uploading checkpoint: {e}")
        return None


def monitor_training(log_file, checkpoint_dir, logger: BaseLogger = None, 
                    interval: int = 15, stall_timeout: int = 600, compress_checkpoints: bool = True):
    """
    Monitor training logs and checkpoints at specified intervals.
    
    Args:
        log_file: Path to the log file to monitor
        checkpoint_dir: Directory containing checkpoints
        logger: Logger to use for tracking
        interval: How often to check for updates (seconds)
        stall_timeout: How long to wait with no changes before stopping (seconds)
        compress_checkpoints: Whether to compress checkpoints before uploading
    """
    fetcher = LogFetcher(log_file, logger=logger, checkpoint_dir=checkpoint_dir)
    
    last_activity_time = time.time()
    had_activity = False
    
    try:
        while True:
            current_time = time.time()
            activity_detected = False
            
            # Fetch and process new logs
            new_logs = fetcher.fetch_new_logs()
            if new_logs:
                log_metrics(logger, new_logs)
                activity_detected = True
                last_activity_time = current_time
                if not had_activity:
                    had_activity = True
                    logging.info("Initial activity detected")
            
            # Detect and process new checkpoints
            new_checkpoint = fetcher.detect_new_checkpoints()
            if new_checkpoint:
                checkpoint_name = upload_checkpoint(
                    logger, new_checkpoint, 
                    cloud_storage_path=None, 
                    compress=compress_checkpoints
                )
                if logger:
                    logger.log_metric("new_checkpoint", 1.0)
                    logger.update_summary("latest_checkpoint", checkpoint_name)
                logging.info(f"New checkpoint processed: {checkpoint_name}")
                activity_detected = True
                last_activity_time = current_time

            had_activity = activity_detected
            
            # Check for stall condition
            if not had_activity and (current_time - last_activity_time) > stall_timeout:
                if logger:
                    logger.log_metric("training_stalled", 1.0)
                    logger.update_summary("stalled_after_seconds", int(current_time - last_activity_time))
                logging.warning(f"Training appears stalled - no activity for {stall_timeout} seconds. Stopping monitoring.")
                break
            
            # Log stall status periodically if we're getting close to timeout
            if not had_activity and (current_time - last_activity_time) > (stall_timeout / 2):
                inactive_time = int(current_time - last_activity_time)
                
                logging.warning(f"No activity detected for {inactive_time} seconds (timeout: {stall_timeout})")
                if logger and inactive_time % 60 == 0:  # Log every minute when getting close to stalled
                    logger.log_metric("inactive_seconds", inactive_time)
            
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logging.error("Monitoring stopped by user")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configure logger based on environment variable or command line arg
    tracking_backend = os.getenv("TRACKING_BACKEND", "wandb")
    logger = create_logger(tracking_backend)
    logger.login()
    
    run = logger.init_run(
        project=os.getenv("WANDB_PROJECT") if tracking_backend == "wandb" else os.getenv("MLFLOW_EXPERIMENT_NAME"),
        entity=os.getenv("WANDB_ENTITY") if tracking_backend == "wandb" else None,
        job_type="training_monitor"
    )
    
    try:
        # Example usage
        monitor_training("outputs/training_log.json", "outputs/checkpoints", logger=logger)
    finally:
        logger.finish_run()