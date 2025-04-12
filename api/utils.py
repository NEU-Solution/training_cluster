import yaml
from fastapi import HTTPException

from const import DEFAULT_CONFIG

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config_from_yaml(yaml_path):
    """Load configuration from a YAML file."""
    try:
        with open(yaml_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        # Start with default config and update with loaded values
        config = DEFAULT_CONFIG.copy()
        config.update(config_dict)
        return config
    except Exception as e:
        logger.error(f"Error loading config from {yaml_path}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load configuration: {str(e)}"
        )