import os
import sys
from src.exception import CustomException
#from src.utils import Utils
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion initiated")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Data ingestion completed")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and Test data saved")

            logging.info("Ingestion of Data is completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path,
                    self.ingestion_config.raw_data_path)
        except Exception as e:
            logging.error("Data ingestion failed")
            raise CustomException("Data ingestion failed", e)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data, raw_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_arr,_ =  data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array, test_arr))
