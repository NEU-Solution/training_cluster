import os
import wandb
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from .base_logger import BaseLogger
import datetime

class WandbLogger(BaseLogger):
    """Weights & Biases implementation of BaseLogger."""
    
    def __init__(self):
        self.run = None
        self.api_key = os.getenv("WANDB_API_KEY")
        self.project = os.getenv("WANDB_PROJECT")
        self.entity = os.getenv("WANDB_ENTITY")
        self.tracking_backend = "wandb"
        
    def login(self, **kwargs):
        """Login to WandB."""
        key = kwargs.get('key', self.api_key)
        wandb.login(key=key)
        
    def init_run(self, project: str = None, entity: str = None, job_type: str = "experiment", 
                 config: Dict[str, Any] = None, name: Optional[str] = None) -> Any:
        """Initialize a new WandB run."""
        self.run = wandb.init(
            project=project or self.project,
            entity=entity or self.entity,
            job_type=job_type,
            config=config,
            name=name
        )
        return self.run
    
    def auto_init_run(self, config = None):
        """Automatically initialize a WandB run with default parameters."""
        project = os.getenv("WANDB_PROJECT", "training")
        entity = os.getenv("WANDB_ENTITY", None)
        job_type = os.getenv("WANDB_JOB_TYPE", "training")
        config = config or {}

        run_name = f"training_{self.training_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        return self.init_run(
            project=project,
            entity=self.entity,
            job_type=job_type,
            config=config,
            name=run_name
        )
    
    def log_metric(self, key: str, value: Union[float, int]) -> None:
        """Log a single metric to WandB."""
        self.run.log({key: value})
        
    def log_metrics(self, metrics: Dict[str, Union[float, int]]) -> None:
        """Log multiple metrics to WandB."""
        self.run.log(metrics)
    
    def log_table(self, key: str, dataframe: pd.DataFrame) -> None:
        """Log a dataframe as a table to WandB."""
        table = wandb.Table(dataframe=dataframe)
        self.run.log({key: table})
    
    def log_artifact(self, local_path: str, name: Optional[str] = None) -> None:
        """Log an artifact file to WandB."""
        artifact = wandb.Artifact(name=name or os.path.basename(local_path), type="dataset")
        artifact.add_file(local_path)
        self.run.log_artifact(artifact)

    def log_artifact(self, local_path: str, name: Optional[str] = None) -> None:
        """Log an artifact file to WandB."""
        artifact = wandb.Artifact(name=name or os.path.basename(local_path), type="file")
        artifact.add_file(local_path)
        self.run.log_artifact(artifact)
        
    def update_summary(self, key: str, value: Any) -> None:
        """Update a summary metric in WandB."""
        self.run.summary[key] = value
    
    def finish_run(self) -> None:
        """End the current WandB run."""
        if self.run:
            self.run.finish()
            self.run = None

    def get_tracking_url(self) -> Optional[str]:
        """Get URL to the current run in WandB UI."""
        if self.run:
            return self.run.get_url()
        return None
    
    def register_model(self, model_path: str, model_name: str, collection_name: str = None, 
                  registry_name: str = "model", **kwargs) -> Optional[str]:
       
        if not self.run:
            print("Cannot register model: No active WandB run")
            return None
            
        try:
            # Log the model as an artifact
            model_artifact = self.run.log_artifact(
                artifact_or_path=model_path,
                name=model_name,
                type="model"
            )
            
            # Link to the registry using the prescribed format
            if collection_name:
                registry_path = f"wandb-registry-{registry_name}/{collection_name}"
                self.run.link_artifact(
                    artifact=model_artifact,
                    target_path=registry_path,
                )
                print(f"Model registered at: {registry_path}")
                
                # Update summary with model registration info
                self.update_summary("registered_model", registry_path)
                self.log_metric("model_registered", 1.0)
                
                return registry_path
            else:
                print("Model artifact created but not registered (no collection_name provided)")
                return model_artifact.name
                
        except Exception as e:
            print(f"Error registering model: {str(e)}")
            return None