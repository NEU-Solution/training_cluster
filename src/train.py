import os
import sys
sys.path.append('..')


from src.collecting_data import load_data
from src.preprocess import create_training_yaml, download_model_regristry
from src.training_cli import TrainingRunner
from src.exp_logging import create_logger, BaseLogger
import random
import string

current_dir = os.path.dirname(os.path.abspath(__file__))

def train(
    model_name: str,
    dataset_version: str,
    cutoff_len: int,
    max_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    logger: BaseLogger = None,
    template: str = "qwen",
    training_type: str = "sft",
    learning_rate: str = '2.0e-4',
    num_epochs: int = 2.0,
    save_steps: int = 1000,
    lora_name: str = None,
    lora_version: str = None,
    lora_hf_repo: str = None,
    tracking_backend: str = "wandb",
    adapter_path: str = None,
    save_name: str = None,
    **kwargs
):
    # Pull data from database
    dataset_name = load_data(dataset_version)
    return
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    
    # Relative to the LLama-Factory directory
    output_dir = f"saves/models/lora/{training_type}/{random_suffix}"

    adapter_dir =f"models/lora"

    if logger is None:
        logger = create_logger(tracking_backend)

    runner = TrainingRunner(
        output_dir= output_dir,
        logger=logger,
    )

    if save_name is None:
        save_name = lora_name

    training_args = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "lora_name": lora_name,
        "lora_version": lora_version,
        'save_name': save_name,
    }

    # Start logging
    runner.start_logging(training_args=training_args)

    # Update logger config
    logger.update_config(training_args)

    if lora_name:
        # Download LoRA weights
        adapter_path = download_model_regristry(
            model_name=lora_name,
            version=lora_version,
            download_dir=adapter_dir,
            logger=logger,
            hf_repo=lora_hf_repo
        )

    # Create training yaml
    yaml_path = create_training_yaml(
        model_name_or_path=model_name,
        dataset_names=dataset_name,
        template=template,
        cutoff_len=cutoff_len,
        max_samples=max_samples,
        batch_size=batch_size,
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir=output_dir,
        adapter_name_or_path=adapter_path,
        stage=training_type,
        **kwargs
    )
    # Create training runner

    # Start training

    llamafactory_path = os.path.join(current_dir, "../LLaMA-Factory")

    cmd = f"cd {llamafactory_path} && llamafactory-cli train {yaml_path}"
    # print(cmd)
    runner.run_training(cmd)


if __name__ == "__main__":
    # Example usage
    train(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_version="v1.0",
        template="qwen",
        cutoff_len=2048,
        max_samples=50000,
        save_steps=200,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate='2.0e-5',
        num_epochs=2.0,
        tracking_backend="mlflow",
        lora_name="initial-sft",
        save_name='test_sft'
    )
