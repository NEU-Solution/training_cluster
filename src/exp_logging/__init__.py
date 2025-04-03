from .base_logger import BaseLogger
from .wandb_logger import WandbLogger
from .mlflow_logger import MLflowLogger

def create_logger(backend: str = "wandb") -> BaseLogger:
    """
    Factory method to create the appropriate logger.
    
    Args:
        backend: String identifier for the logger backend ('wandb' or 'mlflow')
        
    Returns:
        A logger instance implementing the BaseLogger interface
    """
    if backend.lower() == "wandb":
        return WandbLogger()
    elif backend.lower() == "mlflow":
        return MLflowLogger()
    else:
        raise ValueError(f"Unsupported logging backend: {backend}. Supported backends: 'wandb', 'mlflow'")