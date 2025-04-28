import os
import mlflow
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from .base_logger import BaseLogger
import datetime
import logging

from dotenv import load_dotenv
load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

import re

class MLflowLogger(BaseLogger):
    """MLflow implementation of BaseLogger."""
    
    def __init__(self, model_name: str = None, lora_name: str = None):
        
        super().__init__(model_name=model_name, lora_name=lora_name)
        
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "training")
        self.run_id = None
        self.experiment_id = None
        self.tracking_backend = "mlflow"

        self.config = {}

        
        
    def login(self, **kwargs):
        """Set up MLflow tracking."""
        tracking_uri = kwargs.get('tracking_uri', self.tracking_uri)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
    def init_run(self, project: str = None, entity: str = None, job_type: str = "experiment",
                 config: Dict[str, Any] = {}, name: Optional[str] = None) -> Any:
        """Initialize a new MLflow run. If a run is already active, it will be reused. Always update the config."""
        
        # Start the run
        tags = {"job_type": job_type}
        if entity:
            tags["entity"] = entity

        self.update_config(config)

        if 'model_name' in config:
            self.model_name = config['model_name']
        if 'lora_name' in config:
            self.lora_name = config['lora_name']
        if 'save_name' in config:
            self.save_name = config['save_name']
        else:
            self.save_name = config.get('lora_name', self.lora_name)


        if self.run is not None or self.run_id is not None:
            logging.warning("Run already initialized. Please finish the current run before starting a new one.")
            return self.run
        
        # Use project as experiment name if provided
        experiment_name = project or self.experiment_name
        
        # Get or create the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        self.experiment_id = experiment_id

            
        active_run = mlflow.start_run(
                run_name=name,
                experiment_id=experiment_id,
                tags=tags
            )

        self.run = mlflow.tracking.MlflowClient()
        self.run_id = active_run.info.run_id
        
        # Log the config as parameters
        if config:
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)

        logging.info(f"MLflow run initialized with ID: {self.run_id}")    
        return self.run
    

    def auto_init_run(self, config = None):
        project = os.getenv("MLFLOW_EXPERIMENT_NAME", "training")
        job_type = os.getenv("MLFLOW_JOB_TYPE", "training")
        config = config or {}
        run_name = f"auto_run_{self.run_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
        self.init_run(
            project=project,
            job_type=job_type,
            config=config,
            name=run_name
        )

    
    def log_metric(self, key: str, value: Union[float, int], **kwargs) -> None:
        """Log a single metric to MLflow."""
        if not self.check_run_status():
            return
        
        self.run.log_metric(run_id = self.run_id, key = key, value=value, **kwargs)
        
    def log_metrics(self, metrics: Dict[str, Union[float, int]]) -> None:
        """Log multiple metrics to MLflow."""
        if not self.check_run_status():
            return
        
        
        # print(f"Logging metric {key}: {value}")
        step = metrics.pop('current_steps', 0)
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)
    
    def log_table(self, key: str, dataframe: pd.DataFrame) -> None:
        """Log a dataframe as a table to MLflow."""
        # Save dataframe to CSV and log it
        if not self.check_run_status():
            return
        
        temp_path = f"/tmp/{key}.csv"
        dataframe.to_csv(temp_path, index=False)
        self.log_artifact(temp_path, f"tables/{key}")
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
    
    def log_artifact(self, local_path: str, name: Optional[str] = None, type_ = "file") -> str:
        """Log an artifact file to MLflow and return the artifact path.
        
        Args:
            local_path: Path to the local file to log
            name: Optional artifact path to log to within the run
            type_: Type of artifact (default: "file")
            
        Returns:
            String path to the logged artifact in MLflow
        """
        if not self.check_run_status():
            return None
        
        artifact_path = name or ""
        self.run.log_artifact(run_id = self.run_id, local_path = local_path, artifact_path = artifact_path)
        
        # Construct the artifact path in MLflow
        file_name = os.path.basename(local_path)
        if artifact_path:
            full_artifact_path = f"{artifact_path}/{file_name}"
        else:
            full_artifact_path = file_name
            
        return full_artifact_path
    

    def log_artifacts(self, local_path: str, name: Optional[str] = None, type_ = "file") -> str:
        """Log an artifact file to MLflow and return the artifact path.
        
        Args:
            local_path: Path to the local file to log
            name: Optional artifact path to log to within the run
            type_: Type of artifact (default: "file")
            
        Returns:
            String path to the logged artifact in MLflow
        """
        if not self.check_run_status():
            return None
        
        artifact_path = name or ""
        self.run.log_artifacts(run_id = self.run_id, local_dir = local_path, artifact_path = artifact_path)
        
        # Construct the artifact path in MLflow
        file_name = os.path.basename(local_path)
        if artifact_path:
            full_artifact_path = f"{artifact_path}/{file_name}"
        else:
            full_artifact_path = file_name
            
        return full_artifact_path

        
    def update_summary(self, key: str, value: Any) -> None:
        """Update a summary metric in MLflow."""
        # MLflow doesn't have a direct equivalent to wandb's summary
        # We can use log_metric instead
        self.log_metric(f"summary_{key}", value)
    
    def finish_run(self) -> None:
        """End the current MLflow run."""
        logging.info(f"Finishing MLflow run with ID: {self.run_id}")
        mlflow.end_run()
        self.run = None
        self.run_id = None

    def get_tracking_url(self) -> Optional[str]:
        """Get URL to the current run in MLflow UI."""
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri and self.run_id:
            # For hosted MLflow or local server
            if tracking_uri.startswith('http'):
                return f"{tracking_uri}/#/experiments/{self.experiment_id}/runs/{self.run_id}"
        return None
    
    def register_model(self, model_path: str, model_name: str = None, tags: Dict[str, Any] = None, collection_name: str = None,
                       registry_name: str = "model", **kwargs) -> Optional[str]:
        """Register a model with MLflow Model Registry and optionally add it to a collection.
        
        Args:
            model_path: Path to the model to register
            model_name: Name to register the model under (defaults to self.model_name if not pr
            ovided)
            tags: Optional dictionary of tags to add to the model
            collection_name: Optional name of the collection to register the model in
            
        Returns:
            The model version URI after registration
        """
        if not self.check_run_status():
            return None
        
        # Use provided model_name or fall back to instance model_name
        base_name = collection_name or self.model_name
        save_name = self.save_name
        
        if not base_name:
            raise ValueError("Model name must be provided either in method call or at logger initialization")
        
        checkpoint_name = os.path.basename(model_path)

        # First log the model as an artifact
        self.log_artifacts(model_path, name=f"model/{checkpoint_name}")
        
        # Register the model using the logged artifact
        model_uri = f"runs:/{self.run_id}/model/{checkpoint_name}"
        print(f"Registering model from {model_uri} with name {save_name}")
        # Register model with provided URI and name
        result = mlflow.register_model(model_uri=model_uri, name=save_name)
        
        # Add tags to the registered model version if provided
        
        pattern = r"checkpoint-([0-9]+)"
        number = re.search(pattern, checkpoint_name)
        if number:
            checkpoint_number = int(number.group(1))
            self.run.set_model_version_tag(name=save_name, version=result.version, 
                               key="checkpoint", value=checkpoint_number)

        if self.original_version.isdigit():
            self.run.set_model_version_tag(name=save_name, version=result.version, 
                               key="original", value=f"{base_name}-v{self.original_version}")

        self.run.set_model_version_tag(name=save_name, version=result.version, 
                                        key="evaluate", value=f"pending")

        # # Set an alias for the model version if provided in kwargs
        # if 'alias' in kwargs:
        #     alias = kwargs.get('alias')
        #     self.run.set_registered_model_alias(name=name, alias=alias, version=result.version)
        
        
        return f"models:/{save_name}/{result.version}"
        