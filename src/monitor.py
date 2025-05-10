# Automatically fetch json logs, record new logs and log them to tracking systems

import json
import time
import os
from pathlib import Path
import shutil
import datetime
import sys
from typing import Dict, Any, List, Optional
import time
import requests
from requests.exceptions import RequestException
import threading

import logging
logging.basicConfig(level=logging.INFO)

import sys 
sys.path.append('..')

# Import logger classes
from src.exp_logging import WandbLogger, MLflowLogger, BaseLogger, create_logger

from dotenv import load_dotenv
load_dotenv()


from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union


class EvaluationRequest(BaseModel):
    base_model_name: str
    lora_model_name: str
    data_version: str = "latest"
    lora_version: Optional[str] = "latest"
    multi_thread: bool = True
    llm_backend: str = 'vllm'
    tracking_backend: str = 'mlflow'
    train_id: Optional[str] = None



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
            logging.warning("Checkpoint directory does not exist or is not specified.")
            return []
        
        # Check for new checkpoints
        logging.info(f"Checking for new checkpoints in {self.checkpoint_dir}")
        
            
        checkpoints = [
            cp for cp in self.checkpoint_dir.glob("checkpoint-*") 
            if cp.is_dir() and cp not in self.known_checkpoints
        ]
        
        if not checkpoints:
            return []
            
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x.stat().st_mtime,)

        newest_checkpoints = []

        # Need to add condition of having safetensors
        for checkpoint in checkpoints:
            if checkpoint not in self.known_checkpoints:
                newest_checkpoints.append(checkpoint)
                self.known_checkpoints.add(checkpoint)

            return newest_checkpoints
        
        return []


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
            logging.info(f"Logged metrics: {metrics}")
            
        # # Log any non-numeric values as separate artifacts if needed
        # for key, value in log_entry.items():
        #     if not isinstance(value, (int, float)) and key not in ('step', 'timestamp'):
        #         # For complex objects, we could store them as JSON
        #         if isinstance(value, dict) or isinstance(value, list):
        #             logger.log_metric(f"{key}_present", 1.0)



def send_evaluation_request(base_model_name: str, 
                            lora_model_name: str,
                            data_version: str = "latest", 
                            tracking_backend: str = 'mlflow', 
                            eval_server_url: Optional[str] = None,
                            train_id: Optional[str] = None,
                            **kwargs: Dict[str, Any]
                            ):
    """
    Send an API request to the evaluation server for a new model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        model_name: Name of the model/checkpoint
        eval_server_url: URL of the evaluation server (defaults to environment variable)
    """
    if not eval_server_url:
        eval_server_url = os.getenv("EVAL_SERVER_URL", "http://localhost:23477") + "/evaluate"
    


    payload_params = {
        "base_model_name": base_model_name,
        "lora_model_name": lora_model_name,
        "tracking_backend": tracking_backend,
        "data_version": data_version,
        "train_id": train_id,
    }
    # Add any additional parameters from kwargs
    for key, value in kwargs.items():
        if key not in payload_params:
            payload_params[key] = value
    
    # Create the payload
    payload = EvaluationRequest(**payload_params).dict()
    
    def _send_request():
        logging.info(f"Sending evaluation request to {eval_server_url} for {lora_model_name}")
        logging.info(f"Payload: {payload}")
        print ("================= SENDING EVAL REQUEST =================")
        print (payload)
        print ("=========================================================")
        try:
            logging.info(f"Sending evaluation request for checkpoint: {lora_model_name}")
            response = requests.post(eval_server_url, json=payload, timeout=10)
            if response.status_code == 200:
                logging.info(f"Evaluation request accepted for {lora_model_name}")
                return response.json()
            else:
                logging.warning(f"Evaluation server returned status code {response.status_code} for {lora_model_name}")
        except RequestException as e:
            logging.warning(f"Failed to send evaluation request: {e}")
        except Exception as e:
            logging.error(f"Unexpected error sending evaluation request: {e}")
    
    # Run in background thread to avoid blocking
    thread = threading.Thread(target=_send_request)
    thread.daemon = True
    thread.start()



def upload_checkpoint(
    logger: BaseLogger, 
    checkpoint_path: Path,
    cloud_storage_path: str = None, 
    register_to_registry: bool = True,
    collection_name: str = None,
    registry_name: str = "model",
    checkpoint_name: str = None,
    trigger_evaluation: bool = True,
    run_in_background: bool = True
) -> Optional[str]:
    """
    Upload checkpoint to tracking system and optionally to cloud storage
    
    Args:
        logger: BaseLogger instance
        checkpoint_path: Path to the checkpoint directory
        cloud_storage_path: Optional cloud storage path
        register_to_registry: Whether to register the model in the registry
        run_in_background: Whether to run the upload in a background thread
        
    Returns:
        Name of the checkpoint (immediately) or None if invalid parameters
    """
    if not logger:
        logging.warning(f"No logger configured, skipping checkpoint upload: {checkpoint_path}")
        return None
    
    # Default checkpoint name will look like "checkpoint-XXXXX"
    checkpoint_name = checkpoint_path.name if not checkpoint_name else checkpoint_name

    # Delete optimizer.pt file if it exists in the folder
    optimizer_file = Path(os.path.join(checkpoint_path, "optimizer.pt"))
    if optimizer_file.exists():
        logging.info(f"Deleting optimizer file: {optimizer_file}")
        optimizer_file.unlink()

    if not collection_name:
        collection_name = os.getenv("WANDB_REGISTRY", "default")
    
    # Define the upload function to run in background
    def _upload_process():
        try:
            artifact_path = ""

            if register_to_registry:
                # Use WandB model registry feature
                logging.info(f"Registering checkpoint {checkpoint_path} to registry...")
                artifact_path = logger.register_model(
                    model_path=str(checkpoint_path),
                    model_name=checkpoint_name,
                    collection_name=collection_name,
                    registry_name=registry_name
                )
                
                logging.info(f"Model registered at: {artifact_path}")
            else:
                # Log the raw directory as an artifact
                logging.info(f"Uploading raw checkpoint directory {checkpoint_path}...")
                artifact_path = logger.log_directory(str(checkpoint_path), 
                                                    name=f"{logger.config.get('lora_name','')}-{checkpoint_name}", 
                                                    type_="model")
                logging.info(f"Raw checkpoint {logger.config.get('lora_name','')}-{checkpoint_name} uploaded to tracking system")

            # Additional cloud storage upload if needed
            if cloud_storage_path:
                logging.info(f"Uploading checkpoint to cloud storage: {cloud_storage_path}")
                # Cloud upload implementation would go here

            # Trigger evaluation if needed
            if trigger_evaluation and logger.config.get("model_name"):
                print("================= TRIGGERING EVAL =================")
                send_evaluation_request(
                    base_model_name=logger.config.get("model_name"),
                    lora_model_name=artifact_path,
                    data_version=logger.config.get("data_version", "latest"),
                    tracking_backend=logger.tracking_backend,
                    train_id=logger.run_id,
                )
                
            logger.log_metric("checkpoint_upload_complete", 1.0)
            
        except Exception as e:
            logging.error(f"Error uploading checkpoint: {e}")

    # Run in background or foreground based on parameter
    if run_in_background:
        thread = threading.Thread(target=_upload_process)
        thread.daemon = True
        thread.start()
        logging.info(f"Started background upload for checkpoint: {checkpoint_name}")
    else:
        _upload_process()
    
    return checkpoint_name


def scrape_log(fetcher: LogFetcher, **kwargs):
    new_logs = fetcher.fetch_new_logs()
    if new_logs:
        log_metrics(fetcher.logger, new_logs)

    new_checkpoints = fetcher.detect_new_checkpoints()
    for new_checkpoint in new_checkpoints:
    
        checkpoint_name = upload_checkpoint(
            fetcher.logger, new_checkpoint, 
            **kwargs,
        )


def monitor_training(
    fetcher: LogFetcher, 
    interval: int = 15, 
    stall_timeout: int = 180,
    training_completed_event=None,
    upload_timeout: int = 300,  # 5 minutes to complete uploads after training
    trigger_evaluation: bool = True
):    
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
    # fetcher = LogFetcher(log_file, logger=logger, checkpoint_dir=checkpoint_dir)
    

    logger = fetcher.logger
    if not logger:
        logging.warning("No logger configured, skipping monitoring")
        return
    
    max_not_update = int(stall_timeout / interval) * 8
    max_not_update_after_training = int(upload_timeout / interval)
    not_update_count = 0
    
    had_activity = False

    try:
        while True:
            
            activity_detected = False
            
            # Fetch and process new logs
            new_logs = fetcher.fetch_new_logs()
            if new_logs:
                log_metrics(logger, new_logs)
                activity_detected |= True
            
            # Detect and process new checkpoints
            new_checkpoints = fetcher.detect_new_checkpoints()
            for new_checkpoint in new_checkpoints:
                checkpoint_name = upload_checkpoint(
                    logger, new_checkpoint, 
                    cloud_storage_path=None, 
                    trigger_evaluation=trigger_evaluation,
                    run_in_background=True
                )
                logger.log_metric("new_checkpoint", 1.0)
                logging.info(f"New checkpoint processed: {checkpoint_name}")
                activity_detected |= True
            

            # If not started training yet, increase the count for stall condition
            had_activity |= activity_detected

            # If we have had activity, we can reset the stall timeout
            if had_activity:
                max_not_update = int(stall_timeout / interval)
            
            if activity_detected:
                not_update_count = 0
            else:
                not_update_count += 1
                logging.info(f"No updates detected for {not_update_count * interval} seconds")
            # Check for stall condition
            current_max_not_update = max_not_update_after_training if training_completed_event and training_completed_event.is_set() else max_not_update
            
            if not_update_count >= current_max_not_update:
                if training_completed_event and training_completed_event.is_set():
                    logging.info("Training has completed and upload timeout reached, stopping monitoring")
                else:
                    logging.info("Stall condition detected, stopping monitoring")
                break
            
            # Log stall status periodically if we're getting close to timeout
            if not_update_count >= current_max_not_update // 2:
                if training_completed_event and training_completed_event.is_set():
                    logging.info(f"Post-training upload: {not_update_count * interval} seconds without updates")
                else:
                    logging.info(f"Stall condition approaching: {not_update_count * interval} seconds without updates")
            
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
        fetcher = LogFetcher(
            log_file_path="../LLaMA-Factory/saves/models/lora/sft/DS0gpA/trainer_log.jsonl",
            logger=logger,
            checkpoint_dir="../LLaMA-Factory/saves/models/lora/sft/DS0gpA"
        )
        # Example usage
        monitor_training(fetcher, compress_checkpoints=False, interval=5, stall_timeout=60)
    finally:
        logger.finish_run()