import os
import sys
sys.path.append('..')


from src.collecting_data import fake_etl
from src.preprocess import create_training_yaml, download_model_regristry
from src.training_cli import TrainingRunner
from src.exp_logging import create_logger
import random
import string

current_dir = os.path.dirname(os.path.abspath(__file__))

def train(
    model_name: str,
    dataset_version: str,
    template: str,
    cutoff_len: int,
    max_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: str = '2.0e-5',
    num_epochs: int = 2.0,
    save_steps: int = 1000,
    lora_name: str = None,
    lora_version: str = None,
    lora_hf_repo: str = None,
    logging_backend: str = "wandb",
    adapter_path: str = None,
    **kwargs
):
    # Pull data from database
    dataset_name = fake_etl(dataset_version)

    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    
    
    # Relative to the LLama-Factory directory
    output_dir = f"saves/models/lora/sft/{random_suffix}"

    adapter_dir =f"models/lora"

    logger = create_logger(logging_backend)

    runner = TrainingRunner(
        output_dir= output_dir,
        logger=logger,
    )

    training_args = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "lora_name": lora_name,
        "lora_version": lora_version,
    }

    # Start logging
    runner.start_logging(training_args=training_args)

    adapter_path = None
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
        max_samples=1000,
        save_steps=200,
        batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate='2.0e-5',
        num_epochs=3.0,
        logging_backend="wandb",
        lora_name="initial-sft"
    )
