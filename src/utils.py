import os,sys
import pymongo
import pandas as pd

import pickle
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import accuracy_score, classification_report

def export_collection_as_dataframe(url, db_name, collection_name):
    try:
        # connect to data mongodb
        client = pymongo.MongoClient(url)

        # Access the database and collection
        db = client[db_name]
        collection = db[collection_name]

        # Retrive the database 
        cursor = collection.find()
        data_in_json = list(cursor)

        # Convert the data to a Pandas DataFrame
        df = pd.DataFrame(data_in_json)

        # deleting the _id col from dataframe 
        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)
        return df
    except Exception as e:
        raise CustomException(e, sys)

def train_evaluate(x_train, y_train, x_test, y_test, model):
    try:
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        # evaluatiig the model
        score = accuracy_score(y_test, pred)
        report= classification_report(y_test, pred)
        print(f"Accuracy Score: {score * 100:.2f}%")
        print("Report:", report)

        return(model)
    except Exception as e:
        logging.info('Exception occured during model training and Evaluation')
        raise CustomException(e,sys)


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        logging.info('Exception Occured in save_object function utils')
        raise CustomException(e, sys)
    

def load_obj(file_path):
    try:
        with open(file_path, "rb") as file:
            obj = pickle.load(file)
            return obj
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e, sys)