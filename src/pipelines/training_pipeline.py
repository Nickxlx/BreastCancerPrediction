import os, sys

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraininer  

if __name__ == "__main__":
    # data ingestion initialization

    ingestion_obj = DataIngestion()
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()

    print(train_data_path, test_data_path)

    # data transformation 
    transformation_obj = DataTransformation()
    train_arr, test_arr = transformation_obj.initiate_data_transformation(train_data_path, test_data_path)

    # model training 
    model_obj = ModelTraininer()
    model_obj.initiate_model_traininer(train_arr, test_arr)

