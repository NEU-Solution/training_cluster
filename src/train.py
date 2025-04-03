import os
import sys
sys.path.append('..')


from src.collecting_data import fake_etl
from src.preprocess import create_training_yaml
from src.training_cli import TrainingRunner
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
    learning_rate: float,
    num_epochs: int,
    logging_backend: str = "wandb",
    adapter_path: str = None,
):
    # Pull data from database
    dataset_name = fake_etl(dataset_version)

    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    output_dir = f"saves/models/lora/sft/{random_suffix}"


    # Create training yaml
    yaml_path = create_training_yaml(
        model_name_or_path=model_name,
        dataset_names=dataset_name,
        template=template,
        cutoff_len=cutoff_len,
        max_samples=max_samples,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        adapter_path=adapter_path,
        output_dir=output_dir,
    )
    # Create training runner

    

    # Create the training runner
    runner = TrainingRunner(
        output_dir=output_dir,
        tracking_backend=logging_backend,
    )

    # # Start training

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
        batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate='2.0e-5',
        num_epochs=3.0,
        logging_backend="wandb",
    )
