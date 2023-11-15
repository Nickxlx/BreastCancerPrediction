import os, sys
import pandas as pd
import numpy as np

from src.logger import logging 
from src.exception import CustomException
from src.utils import save_obj

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def drop_corr(self, df, threshold=0.9):
        corr_col = set()
        corr_matrix = df.corr()

        for i in range(len(df.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    col_name = corr_matrix.columns[i]
                    corr_col.add(col_name)
                    
        df_dropped = df.drop(columns=corr_col)
        return df_dropped

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transfromation initiated")

            col_to_drop = ('drop_correlated_cols', FunctionTransformer(self.drop_corr, validate=False))
            imputer_step = ('imputer', SimpleImputer(strategy='mean'))
            scaler_step = ('scaler', StandardScaler())

            logging.info("Data Transformation Pipeline Initiated: Drop correlated columns, impute missing values, and scale features.") 
            
            preprocessor = Pipeline(steps=[
                            col_to_drop,
                            imputer_step,
                            scaler_step
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data is completed")

            target_col = "target"

            x_input = train_df.drop(target_col, axis=1)
            y_input = train_df[target_col]
                
            x_test = test_df.drop(target_col, axis=1)
            y_test = test_df[target_col]

            logging.info("Applying preprocessing object on training and testing datasets.")

            preprocessor = self.get_data_transformation_obj()

            x_processed_input = preprocessor.fit_transform(x_input)
            x_processed_test = preprocessor.transform(x_test)

            train_arr = np.c_[x_processed_input, np.array(y_input)]
            test_arr = np.c_[x_processed_test,  np.array(y_test)]

            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                obj = preprocessor)

            logging.info("Preprocessor pickle is created and saved")

            return(train_arr, test_arr)
        
        except Exception as e:
            raise CustomException(e, sys)