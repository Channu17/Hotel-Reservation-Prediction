from src.data_ingestion import DataIngestion
from src.data_preprocessing import Dataprocessor
from src.model_training import ModelTraining
from config.paths_config import *
from utils.common_functions import read_yaml_file



if __name__ == "__main__":
    
    config = read_yaml_file(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()
    
    data_processor = Dataprocessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    data_processor.process()
    
    trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_FILE_PATH,
        test_path=PROCESSED_TEST_FILE_PATH,
        model_output_dir=MODEL_OUTPUT_DIR
    )
    trainer.run()