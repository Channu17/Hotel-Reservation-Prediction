import os
import pandas
from src.logger import get_logger
from src.custom_execption import CustomException   
import yaml


logger = get_logger(__name__)

def read_yaml_file(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        logger.info("Successfully read the YAML file.")
        return config
    except Exception as e:
        logger.error("Error reading the YAML file")
        raise CustomException("Failed to read the YAML file", e)
