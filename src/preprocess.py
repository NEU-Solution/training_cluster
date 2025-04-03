# Convert data to LLaMA-Factory ready format
# Create training yaml file


# Convert data to LLaMA-Factory ready format
# Create training yaml file

import os
import yaml
from pathlib import Path
import tempfile

current_dir = Path(__file__).resolve().parent

def create_training_yaml(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path=None,
    dataset_names=None,
    template="qwen",
    cutoff_len=2048,
    max_samples=10000,
    output_dir="saves/models/lora/sft",
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate='2.0e-5',
    num_epochs=2.0,
    lora_rank=64
):
    """
    Create a YAML configuration file for LLaMA-Factory training.
    
    Args:
        model_name_or_path (str): Path to the base model
        adapter_path (str, optional): Path to LoRA adapter checkpoint
        dataset_names (list, optional): List of dataset names
        template (str): Template for formatting the dataset
        cutoff_len (int): Maximum sequence length
        max_samples (int): Maximum number of training samples
        output_dir (str): Directory to save model outputs
        batch_size (int): Training batch size per device
        gradient_accumulation_steps (int): Number of steps for gradient accumulation
        learning_rate (float): Learning rate for training
        num_epochs (float): Number of training epochs
        lora_rank (int): Rank for LoRA fine-tuning
        temp_dir (str, optional): Directory to save the YAML file (uses system temp if None)
        
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
        "save_steps": 2000,
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
    if adapter_path:
        config["adapter_name_or_path"] = adapter_path
    
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