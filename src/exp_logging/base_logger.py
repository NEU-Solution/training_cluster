from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import logging
import threading


class BaseLogger(ABC):
    """Abstract base class for experiment tracking loggers."""
    _instances = {}
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super(BaseLogger, cls).__new__(cls)
                cls._instances[cls] = instance
            return cls._instances[cls]
    
    @abstractmethod
    def __init__(self, model_name: str = None, lora_name: str = None):
        """
        Initialize the logger.
        
        Args:
            model_name: Name of the model
            lora_name: Name of the LoRA model
        """
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.model_name = model_name
            self.lora_name = lora_name
            self.config = {}
            self.tracking_backend = None
            self.run = None

    @abstractmethod
    def auto_init_run(self):
        pass

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
    def log_artifact(self, local_path: str, name: Optional[str] = None, type_: str = "file") -> None:
        """Log an artifact file."""
        pass

    def log_directory(self, local_dir: str, name: Optional[str] = None, 
                      type_: str = "directory") -> None:
        """
        Log a directory as an artifact.
        By default, uses log_artifact but can be overridden for more efficient implementations.
        """
        return self.log_artifact(local_dir, name, type_=type_)
        
        
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

    def check_run_status(self) -> bool:
        """
        Check if the current run is active.
        
        Returns:
            True if the run is active, False otherwise
        """
        flag = hasattr(self, 'run') and self.run is not None
        if not flag:
            logging.warning("No active run detected.")
        return flag