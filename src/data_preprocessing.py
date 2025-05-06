import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_execption import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml_file, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)   

class Dataprocessor:
    
    def __init__(self, train_path, test_path,processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config_path = read_yaml_file(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
    def preprocess_data(self, df):
        try:
            logger.info("Started our Data preprocessing")
            
            logger.info("dropping the columns") 
            df.drop(columns=['Unnamed: 0', 'index'], inplace=True, errors='ignore')
            df.drop_duplicates(inplace=True)    
            
            cat_cols = self.config_path['data_preprocessing']['categorical_columns']
            num_cols = self.config_path['data_preprocessing']['numerical_columns']
            
            logger.info("Encoding categorical columns")
            label_encoder = LabelEncoder()
            
            label_encoder = LabelEncoder()  
            mappings = {}

            for column in cat_cols: 
                df[column] = label_encoder.fit_transform(df[column])
                mappings[column] = {label:code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

            logger.info("Label encoding completed successfully")
            
            logger.info("doing the skewness transformation")
            skew_threshold = self.config_path['data_preprocessing']['skewness_threshold']
            
            for column in num_cols:
                if abs(df[column].skew()) > skew_threshold:
                    df[column] = np.log1p(df[column])
                    
            logger.info("Skewness transformation completed successfully")
            
            return df
        except Exception as e:
            logger.error(f"Error in data preprocessing {e}")
            raise CustomException("Data preprocessing failed", e)