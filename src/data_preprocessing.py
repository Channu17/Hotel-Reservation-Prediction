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
            df.drop(columns=['Unnamed: 0',"Booking_ID"], inplace=True)
            df.drop_duplicates(inplace=True)    
            
            cat_cols = self.config_path['data_processing']['categorical_columns']
            num_cols = self.config_path['data_processing']['numerical_columns']
            
            logger.info("Encoding categorical columns")
            label_encoder = LabelEncoder()
            
            label_encoder = LabelEncoder()  
            mappings = {}

            for column in cat_cols: 
                df[column] = label_encoder.fit_transform(df[column])
                mappings[column] = {label:code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

            logger.info("Label encoding completed successfully")
            
            logger.info("doing the skewness transformation")
            skew_threshold = self.config_path['data_processing']['skewness_threshold']
            
            for column in num_cols:
                if abs(df[column].skew()) > skew_threshold:
                    df[column] = np.log1p(df[column])
                    
            logger.info("Skewness transformation completed successfully")
            
            return df
        except Exception as e:
            logger.error(f"Error in data preprocessing {e}")
            raise CustomException("Data preprocessing failed", e)
    
    def balance_data(self, df):
        try:
            logger.info("Balancing the data using SMOTE")
            
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['target'] = y_resampled 
            logger.info("Data balancing completed successfully")
            
            return pd.concat([X_resampled, y_resampled], axis=1)
        except Exception as e:
            logger.error(f"Error in data balancing {e}")
            raise CustomException("Data balancing failed", e)
     
    def select_features(self, df):
        try:
            logger.info("Feature selection using Random Forest")
            
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            
            feature_importances = model.feature_importances_

            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importances
            })
            # sort by importance desc
            sorted_df = feature_importance_df.sort_values(
                by='importance', ascending=False
            )
            # pick top N feature names
            num_top = self.config_path['data_processing']['no_of_features']
            top_features = sorted_df.head(num_top)['feature'].tolist()

            # now subset original df by those feature names + target
            top_10_df = df[top_features + ['booking_status']]

            logger.info("Feature selection completed successfully")
            return top_10_df
        except Exception as e:
            logger.error(f"Error in feature selection {e}")
            raise CustomException("Feature selection failed", e)
        
    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving the processed data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info("Data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving the data {e}")
            raise CustomException("Failed to save the data", e)
        
    def process(self):
        try:
            logger.info("loading the data")
            
            train_df = load_data(self.train_path)
            
            test_df = load_data(self.test_path)
            
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
            
            train_df = self.balance_data(train_df)  
            test_df = self.balance_data(test_df)
            
            train_df = self.select_features(train_df)
            
            test_df = test_df[train_df.columns]
            
            self.save_data(train_df, PROCESSED_TRAIN_FILE_PATH)
            self.save_data(test_df, PROCESSED_TEST_FILE_PATH)   
            
            logger.info("Data processing completed successfully")   
        except Exception as e:
            logger.error(f"Error in data processing {e}")
            raise CustomException("Data processing failed", e)
        finally:
            logger.info("Data processing finished")
            
if __name__ == "__main__":
    
    data_processor = Dataprocessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    data_processor.process()

