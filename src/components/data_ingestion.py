import os, sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import export_collection_as_dataframe

from sklearn.model_selection import train_test_split 

from dataclasses import dataclass 

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts", "raw.csv")
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion is Just Started")
        try:
            url = r"mongodb+srv://nikhilsinghxlx:Nikhilsinghxlx@cluster0.9kjhcgg.mongodb.net/?retryWrites=true&w=majority"
            DataBase_Name = "Projects" 
            Collection_Name = "Breast_cancer"

            df = export_collection_as_dataframe(url,
                                                DataBase_Name,
                                                Collection_Name) 
            
            logging.info("Data as DataFrame is Successfully exported")

            # creating folder to store the data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index = False)

            logging.info("Train Test split")
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42 )

            # storing train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header = True)

            logging.info("Data Ingetion is completed")

            return(self.ingestion_config.train_data_path, 
                    self.ingestion_config.test_data_path
                )
        
        except Exception as e:
            raise CustomException(e,sys) 
            