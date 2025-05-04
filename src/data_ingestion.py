import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_execption import CustomException
from utils.common_functions import read_yaml_file
from config.paths_config import *


logger = get_logger(__name__)


class DataIngestion:
    
    def __init__(self, config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.file_name = self.config['bucket_file_name']
        self.train_ratio = self.config['train_ratio']
        
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Data ingestion started with {self.bucket_name} and file name is {self.file_name}")
    
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            
            blob.download_to_filename(RAW_FILE_PATH)
            
            logger.info(f"Raw file is successfully downloaded from GCP bucket {self.bucket_name} to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error("Error downloading the CSV file from GCP bucket")
            raise CustomException("Failed to download the CSV file from GCP bucket", e)
    
    def split_data(self):
        try:
            logger.info("Splitting the data into train and test sets")
            data= pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(data, train_size=self.train_ratio, random_state=42)
            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)
            logger.info(f"Train and test data are successfully split and saved to {TRAIN_FILE_PATH} and {TEST_FILE_PATH}")
            
        except Exception as e:
            logger.error("Error splitting the data into train and test sets")
            raise CustomException("Failed to split the data into train and test sets", e)
        

    def run(self):
        try:
            self.download_csv_from_gcp()
            self.split_data()
            
            logger.info("Data ingestion completed successfully")
        except Exception as e:
            logger.error("Error in data ingestion process")
            raise CustomException("Data ingestion process failed", e)

        finally:
            logger.info("Data ingestion process finished")
    
if __name__ == "__main__":
    
    config = read_yaml_file(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()
    
    
