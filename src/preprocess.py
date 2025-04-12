# Convert data to LLaMA-Factory ready format
# Create training yaml file


# Convert data to LLaMA-Factory ready format
# Create training yaml file

import os
import yaml
from pathlib import Path
import tempfile

import wandb
import mlflow
import logging
from src.exp_logging import BaseLogger
from huggingface_hub import snapshot_download

current_dir = Path(__file__).resolve().parent

def download_model_regristry(model_name: str, version: str = None, download_dir: str = 'models', logger: BaseLogger = None, hf_repo: str = None) -> str:
    """
    Download a model from the WandB model registry.
    """

    assert model_name, "Model name can not be empty"
    assert logger, "No logger instance provided"

    # if 'wandb-registry-model' not in model_name:
    #     model_name = 'wandb-registry-model/' + model_name

    # Initialize a W&B run
    
    # Download the model

    download_dir = os.path.join('../', download_dir)
    os.makedirs(download_dir, exist_ok=True)

    if hf_repo is not None:
        # Download from Hugging Face Hub
        artifact_dir = snapshot_download(
            repo_id=hf_repo,
            revision=version,
            cache_dir=download_dir
        )
        logging.info(f"Downloaded model from Hugging Face Hub to {artifact_dir}")
        
        if version is not None:
            artifact_dir = os.path.join(artifact_dir, version)
        
        return artifact_dir

    if logger.tracking_backend == 'wandb':
        
        if 'wandb-registry-model' not in model_name:
            model_name = 'wandb-registry-model/' + model_name
        
        # Download the model using wandb API
        artifact = wandb.use_artifact(
            f"{model_name}:{version}" if version else f"{model_name}:latest"
        )
        artifact_dir = artifact.download(root=download_dir)

    elif logger.tracking_backend == 'mlflow':
        # Handle MLflow model download
        if version is None:
            version = "latest"
            
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        
        # Download via MLflow
        artifact_dir = os.path.join(download_dir, model_name.replace("/", "_"))
        registered_model = mlflow.register_model(
            f"models:/{model_name}/{version}",
            model_name
        )
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}/{version}",
            dst_path=artifact_dir
        )
    else:
        raise ValueError(f"Unsupported logger")
        
    logging.info(f"Downloaded model {model_name} version {version} to {artifact_dir}")
    
    return artifact_dir

def create_training_yaml(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    adapter_name_or_path=None,
    dataset_names=None,
    template="qwen",
    cutoff_len=2048,
    max_samples=10000,
    output_dir="saves/models/lora/sft",
    save_steps=1000,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate='2.0e-5',
    num_epochs=2.0,
    lora_rank=64
):
    """
    Create a YAML configuration file for LLaMA-Factory training.
    
    Args:

    Returns:
        str: Path to the created YAML file
    """
    if dataset_names is None:
        dataset_names = ["alpaca_en_demo", "alpaca_en_demo"]
    
    # Create config dictionary
    config = {
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": True,
        
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_rank,
        "lora_target": "all",
        
        "dataset": ",".join(dataset_names),
        "template": template,
        "cutoff_len": cutoff_len,
        "max_samples": max_samples,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "dataloader_num_workers": 4,
        
        "output_dir": output_dir,
        "logging_steps": 20,
        "save_steps": save_steps,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        "report_to": "none"

    }
    
    # Add adapter path if provided
    if adapter_name_or_path:
        config["adapter_name_or_path"] = adapter_name_or_path
    
    # Create temp directory if not provided
    os.makedirs(os.path.join(current_dir, '../temp'), exist_ok=True)
    
    # Create file name
    yaml_path = os.path.join(current_dir, '../temp', f"training_config.yaml")
    
    # Write config to file with proper formatting
    with open(yaml_path, 'w') as f:
        # Custom formatting to match the template style
        for key, value in config.items():
            if value is None:  # This is a section header
                f.write(f"{key}: null\n")
            else:
                f.write(f"{key}: {value}\n")
    
    return yaml_path


if __name__ == "__main__":
    # Example usage
    config_path = create_training_yaml(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_names=["your_dataset1", "your_dataset2"],
        template="qwen",
        output_dir="saves/models/custom_output",
        temp_dir='temp'  # Specify your temp directory here
    )
    print(f"Created configuration file at: {config_path}")