from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

## Data Transformation config

@dataclass 
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')


## Data IngestionConfig class
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            
            logging.info('Data transformation initiated')
            #Columns which should be encoded and scaled
            cat_cols=['cut','color','clarity']
            num_cols=['carat','depth','table','x','y','z']

            #Custom ranking for ordinal variable
            cut_cat=['Fair','Good','Very Good','Premium','Ideal']
            color_cat=['D','E','F','G','H','I','J']
            clarity_cat=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline initiated')

            num_pipe=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipe=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OrdinalEncoder(categories=[cut_cat,color_cat,clarity_cat])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipe',num_pipe,num_cols),
                ('cat_pipe',cat_pipe,cat_cols)
            ])

            logging.info('Pipeline Completed')
            return preprocessor
        

        except Exception as e:
            logging.info('Error occured in data transformation pipeline')
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            
            logging.info('Read Train and test data')
            logging.info(f'Train DataFrame Head: \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head: \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformation_object()
            target_column='price'
            drop_col=[target_column,'id']

            input_feature_train_df=train_df.drop(columns=drop_col,axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=drop_col,axis=1)
            target_feature_test_df=test_df[target_column]

            ##applying the transformation

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and test data')

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            logging.info('Exception occured in initiate_dataTransformation')
            raise CustomException(e,sys)