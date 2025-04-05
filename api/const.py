from pydantic import BaseModel, HttpUrl, Field
import os

from typing import Optional, Dict, Any, List, Union
from enum import Enum

# Default configuration (same as in main.py)
DEFAULT_CONFIG = {
    # Model configuration
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset_version": "v1.0",
    "template": "qwen",
    "cutoff_len": 2048,
    "max_samples": 1000,
    "batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": '2.0e-5',
    "num_epochs": 3.0,
    "adapter_path": None,
    
    # Tracking configuration
    "tracking_backend": 'wandb',
    
    # WandB specific config
    "wandb_project": os.getenv("WANDB_PROJECT"),
    "wandb_entity": os.getenv("WANDB_ENTITY"),
    
    # MLflow specific config
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
    "mlflow_experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "model-training"),
}

# Enums for API options
class ConcurrencyStrategy(str, Enum):
    REJECT = "reject"
    QUEUE = "queue"

class TrackingBackend(str, Enum):
    WANDB = "wandb"
    MLFLOW = "mlflow"

# Request model
class TrainingRequest(BaseModel):
    model_name: str = Field(
        default=DEFAULT_CONFIG["model_name"],
        description="Base model to use for training"
    )
    dataset_version: str = Field(
        default=DEFAULT_CONFIG["dataset_version"],
        description="Version of the dataset to use"
    )
    template: str = Field(
        default=DEFAULT_CONFIG["template"],
        description="Template to use for formatting prompts"
    )
    cutoff_len: int = Field(
        default=DEFAULT_CONFIG["cutoff_len"],
        description="Maximum token length for training examples"
    )
    max_samples: int = Field(
        default=DEFAULT_CONFIG["max_samples"],
        description="Maximum number of samples to use from dataset"
    )
    batch_size: int = Field(
        default=DEFAULT_CONFIG["batch_size"],
        description="Batch size for training"
    )
    gradient_accumulation_steps: int = Field(
        default=DEFAULT_CONFIG["gradient_accumulation_steps"],
        description="Number of gradient accumulation steps"
    )
    learning_rate: str = Field(
        default=DEFAULT_CONFIG["learning_rate"],
        description="Learning rate for training"
    )
    num_epochs: float = Field(
        default=DEFAULT_CONFIG["num_epochs"],
        description="Number of training epochs"
    )
    adapter_path: Optional[str] = Field(
        default=None,
        description="Path to adapter/LoRA weights to continue training from"
    )
    tracking_backend: TrackingBackend = Field(
        default=TrackingBackend.WANDB,
        description="Backend to use for experiment tracking"
    )
    webhook_url: Optional[HttpUrl] = Field(
        default=None,
        description="URL to send webhook notifications about job status"
    )

# Response models
class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class TrainingStatus(BaseModel):
    job_id: str
    status: str
    config: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    tracking_url: Optional[str] = None
    error: Optional[str] = None
    output_path: Optional[str] = None