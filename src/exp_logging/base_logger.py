from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, List, Union

class BaseLogger(ABC):
    """Abstract base class for experiment tracking loggers."""
    
    @abstractmethod
    def init_run(self, project: str, entity: str, job_type: str, config: Dict[str, Any] = None, 
                 name: Optional[str] = None) -> Any:
        """Initialize a new run."""
        pass
    
    @abstractmethod
    def log_metric(self, key: str, value: Union[float, int]) -> None:
        """Log a single metric."""
        pass
        
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Union[float, int]]) -> None:
        """Log multiple metrics."""
        pass
    
    @abstractmethod
    def log_table(self, key: str, dataframe: pd.DataFrame) -> None:
        """Log a dataframe as a table."""
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: str, name: Optional[str] = None) -> None:
        """Log an artifact file."""
        pass

    def log_directory(self, local_dir: str, name: Optional[str] = None, 
                      artifact_type: str = "directory") -> None:
        """
        Log a directory as an artifact.
        By default, uses log_artifact but can be overridden for more efficient implementations.
        """
        return self.log_artifact(local_dir, name)
        
        
    @abstractmethod
    def update_summary(self, key: str, value: Any) -> None:
        """Update a summary metric."""
        pass
    
    @abstractmethod
    def finish_run(self) -> None:
        """End the current run."""
        pass
    
    @abstractmethod
    def login(self, **kwargs) -> None:
        """Login to the tracking service."""
        pass

    def get_tracking_url(self) -> Optional[str]:
        """Get URL to the current run in the tracking UI, if available."""
        return None

    def register_model(self, model_path: str, model_name: str, collection_name: str = None, 
                  registry_name: str = "model", **kwargs) -> Optional[str]:
        """
        Register a model to a model registry (if supported).
        
        Args:
            model_path: Path to the model directory or file
            model_name: Name for the model artifact
            collection_name: Collection name for grouping related models
            registry_name: Name of the registry
            **kwargs: Additional backend-specific keyword arguments
            
        Returns:
            Reference to the registered model if successful, None otherwise
        """
        # Default implementation - not supported
        return None