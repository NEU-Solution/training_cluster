from src.train import train
import datetime
import os
import wandb
from types import SimpleNamespace
import argparse
import yaml
from dotenv import load_dotenv
from src.exp_logging import BaseLogger, create_logger
import logging


load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")


# Default configuration
DEFAULT_CONFIG = {
    # Model configuration
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "lora_name": "initial-sft",
    "dataset_version": "v1.0",
    "template": "qwen",
    "cutoff_len": 2048,
    "max_samples": 1000,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": "2.0e-5",
    "num_epochs": 3.0,
    "adapter_path": None,
    "save_steps": 100,
    
    # Tracking configuration
    "tracking_backend": 'wandb',
    
    # WandB specific config
    "wandb_project": os.getenv("WANDB_PROJECT"),
    "wandb_entity": os.getenv("WANDB_ENTITY"),
    
    # MLflow specific config
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
    "mlflow_experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "model-training"),
}


def load_config_from_yaml(yaml_path):
    """Load configuration from a YAML file."""
    with open(yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    
    DEFAULT_CONFIG.update(config_dict)  # Update default config with loaded values

    # Convert dictionary to SimpleNamespace
    return SimpleNamespace(**DEFAULT_CONFIG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified parameters")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        print("Using default configuration")
        config = SimpleNamespace(**DEFAULT_CONFIG)
    
    # Set up logger
    logger = create_logger(config.tracking_backend)
    logger.login()
    
    # Initialize the tracking run
    run_name = f"train_vi_llm_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    if config.tracking_backend == 'wandb':
        wandb.login(key=WANDB_API_KEY)
        run = logger.init_run(
            project=config.wandb_project,
            entity=config.wandb_entity,
            job_type="training",
            config=vars(config),
            name=run_name
        )
    else:  # mlflow
        run = logger.init_run(
            project=config.mlflow_experiment_name,
            job_type="training",
            config=vars(config),
            name=run_name
        )
    
    # Run the training
    print(config.learning_rate)
    try:
        train(
            model_name=config.model_name,
            dataset_version=config.dataset_version,
            template=config.template,
            cutoff_len=config.cutoff_len,
            max_samples=config.max_samples,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=str(config.learning_rate),
            num_epochs=config.num_epochs,
            logging_backend=config.tracking_backend,
            adapter_path=config.adapter_path,
        )
    except KeyboardInterrupt:
        logging.error("Training interrupted. Stopping gracefully...")
        logger.finish_run()
        
    print("Training completed.")