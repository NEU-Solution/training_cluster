import os
import mlflow
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from .base_logger import BaseLogger

class MLflowLogger(BaseLogger):
    """MLflow implementation of BaseLogger."""
    
    def __init__(self):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Default")
        self.run_id = None
        self.tracking_backend = "mlflow"
        
    def login(self, **kwargs):
        """Set up MLflow tracking."""
        tracking_uri = kwargs.get('tracking_uri', self.tracking_uri)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
    def init_run(self, project: str = None, entity: str = None, job_type: str = "experiment",
                 config: Dict[str, Any] = None, name: Optional[str] = None) -> Any:
        """Initialize a new MLflow run."""
        # Use project as experiment name if provided
        experiment_name = project or self.experiment_name
        
        # Get or create the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
            
        # Start the run
        tags = {"job_type": job_type}
        if entity:
            tags["entity"] = entity
        
        active_run = mlflow.start_run(
            run_name=name,
            experiment_id=experiment_id,
            tags=tags
        )
        self.run_id = active_run.info.run_id
        
        # Log the config as parameters
        if config:
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                    
        return active_run
    
    def log_metric(self, key: str, value: Union[float, int]) -> None:
        """Log a single metric to MLflow."""
        mlflow.log_metric(key, value)
        
    def log_metrics(self, metrics: Dict[str, Union[float, int]]) -> None:
        """Log multiple metrics to MLflow."""
        mlflow.log_metrics(metrics)
    
    def log_table(self, key: str, dataframe: pd.DataFrame) -> None:
        """Log a dataframe as a table to MLflow."""
        # Save dataframe to CSV and log it
        temp_path = f"/tmp/{key}.csv"
        dataframe.to_csv(temp_path, index=False)
        mlflow.log_artifact(temp_path, f"tables/{key}")
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
    
    def log_artifact(self, local_path: str, name: Optional[str] = None) -> None:
        """Log an artifact file to MLflow."""
        artifact_path = name or ""
        mlflow.log_artifact(local_path, artifact_path)

    def log_directory(self, local_dir: str, name: Optional[str] = None, 
                     artifact_type: str = "directory") -> None:
        """
        Log a directory as an artifact to MLflow.
        In MLflow, we use log_artifacts for directories.
        """
        artifact_path = name or os.path.basename(local_dir)
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
        
    def update_summary(self, key: str, value: Any) -> None:
        """Update a summary metric in MLflow."""
        # MLflow doesn't have a direct equivalent to wandb's summary
        # We can use log_metric instead
        self.log_metric(f"summary_{key}", value)
    
    def finish_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        self.run_id = None

    def get_tracking_url(self) -> Optional[str]:
        """Get URL to the current run in MLflow UI."""
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri and self.run_id:
            # For hosted MLflow or local server
            if tracking_uri.startswith('http'):
                return f"{tracking_uri}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{self.run_id}"
        return None