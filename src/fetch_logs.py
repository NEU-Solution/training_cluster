# Automatically fetch json logs, record new logs and log it to wandb

import json
import time
import os
from pathlib import Path
import shutil
import datetime


class LogFetcher:
    def __init__(self, log_file_path, checkpoint_dir=None, last_line=0):
        self.log_file_path = Path(log_file_path)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.last_line = last_line
        self.last_checkpoint = None
        self.known_checkpoints = set()
        
    def fetch_new_logs(self):
        """Fetch new logs from the training process."""
        if not self.log_file_path.exists():
            return []
        
        try:
            with open(self.log_file_path, 'r') as f:
                log_data = json.load(f)
                
            logs = log_data.get("logs", [])
            new_logs = logs[self.last_line:]
            self.last_line = len(logs)
            return new_logs
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading log file: {e}")
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


def log_to_wandb(logs):
    """Send logs to Weights & Biases."""
    # Placeholder for actual W&B integration
    # In reality, you'd use:
    # import wandb
    # wandb.log(logs)
    print(f"Logging to W&B: {logs}")


def upload_checkpoint(checkpoint_path, cloud_storage_path):
    """Upload checkpoint to cloud storage."""
    # Placeholder for actual cloud upload
    # In a real implementation, you'd use a cloud SDK
    # For example, for AWS S3:
    # import boto3
    # s3 = boto3.client('s3')
    # s3.upload_file(str(checkpoint_path), 'my-bucket', f'{cloud_storage_path}/{checkpoint_path.name}')
    print(f"Uploading checkpoint {checkpoint_path} to {cloud_storage_path}")
    return f"{cloud_storage_path}/{checkpoint_path.name}"


def monitor_training(log_file, checkpoint_dir, interval=15):
    """Monitor training logs and checkpoints at specified intervals."""
    fetcher = LogFetcher(log_file, checkpoint_dir)
    
    try:
        while True:
            # Fetch and process new logs
            new_logs = fetcher.fetch_new_logs()
            if new_logs:
                log_to_wandb(new_logs)
            
            # Detect and process new checkpoints
            new_checkpoint = fetcher.detect_new_checkpoints()
            if new_checkpoint:
                cloud_path = upload_checkpoint(new_checkpoint, "cloud-storage/checkpoints")
                print(f"New checkpoint uploaded to {cloud_path}")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitoring stopped")


if __name__ == "__main__":
    # Example usage
    monitor_training("outputs/training_log.json", "outputs/checkpoints")