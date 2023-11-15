import os, sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import train_evaluate, save_obj

from sklearn.linear_model import LogisticRegression

from dataclasses import dataclass

class ModelTraininerConfig:
    trained_model_path = os.path.join("artifacts","model.pkl")

class ModelTraininer:
    def __init__(self):
        self.model_trainer_config = ModelTraininerConfig()

    def initiate_model_traininer(self, train_arr, test_arr):
        try:
            logging.info("Splitting the variables into independent and dependent vaiables")

            x_train, x_test, y_train, y_test = (train_arr[:,:-1], test_arr[:,:-1], train_arr[:,-1], test_arr[:,-1])

            logging.info("Training the model")

            model = LogisticRegression()
            
            model = train_evaluate(x_train, y_train, x_test, y_test, model)

            logging.info("Saving the model")

            save_obj(
                file_path=self.model_trainer_config.trained_model_path,
                obj = model
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
            



