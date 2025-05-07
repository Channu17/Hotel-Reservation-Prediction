import os
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_execption import CustomException    
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml_file, load_data
from scipy.stats import randint

logger = get_logger(__name__)

class ModelTraining:
    
    def __init__(self, train_path, test_path, model_output_dir):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_dir = model_output_dir
        self.params_dict = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS    
    
    def load_and_split_file(self):
        try:
            logger.info(f"Loading data from {self.train_path} and {self.test_path}")
            train_df = pd.read_csv(self.train_path) 
            test_df = pd.read_csv(self.test_path)
            
            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']
            
            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']
            
            logger.info("Data loaded and split into features and target variable")
            
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error in loading and splitting data: {e}")
            raise CustomException("Failed to load data",e)
    
    def train_lgbm(self, X_train, y_train):
        try: 
            logger.info("Training LGBM model")
            lgbm_model = LGBMClassifier(random_state=42)
            logger.info("starting the hyperparameter tuning")   

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dict,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )
            logger.info("starting the model training")
            random_search.fit(X_train, y_train)
            
            logger.info("Model training completed successfully")
            
            best_params = random_search.best_params_    
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best parameters: {best_params}")
            
            return best_lgbm_model
        except Exception as e:
            logger.error(f"Error in training LGBM model: {e}")
            raise CustomException("Model training failed", e)

    def evaluate_model(self, model, X_test, y_test):
        
        try: 
            logger.info("Evaluating the model")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")
            logger.info(f"Model evaluation completed successfully")
            
            return {
                'accuracy': accuracy_score,
                'precision': precision,
                'recall': recall_score,
                'f1_score': f1
            }
        except Exception as e:
            logger.error(f"Error in evaluating the model: {e}")
            raise CustomException("Model evaluation failed", e)
        
    def save_model(self, model):
        try:
            logger.info(f"saving the model to {self.model_output_dir}")
            os.makedirs(os.path.dirname(self.model_output_dir), exist_ok=True)
            joblib.dump(model, self.model_output_dir)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error in saving the model: {e}")
            raise CustomException("Model saving failed", e)
    
    
    def run(self):
        try:
            logger.info("Starting the model training process")
            
            X_train, y_train, X_test, y_test = self.load_and_split_file()
            
            model = self.train_lgbm(X_train, y_train)
            
            evaluation_metrics = self.evaluate_model(model, X_test, y_test)
            
            self.save_model(model)
            
            logger.info("Model training process completed successfully")
            
            return evaluation_metrics
        except Exception as e:
            logger.error(f"Error in model training process: {e}")
            raise CustomException("Model training process failed", e)

if __name__ == "__main__":
    trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_FILE_PATH,
        test_path=PROCESSED_TEST_FILE_PATH,
        model_output_dir=MODEL_OUTPUT_DIR
    )
    trainer.run()
    