import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging
from src.exception import CustomException

from src.utils import save_obj
from src.utils import eval_model

from dataclasses import dataclass
import sys 
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_filepath=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting dataset into train and test of dependent and independent')
            Xtr,Xte,ytr,yte=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )

            logging.info('Successfully splitted')

            models={
                'linear_reg':LinearRegression(),
                'lasso':Lasso(),
                'ridge':Ridge(),
                'elas':ElasticNet(),
                'dtree':DecisionTreeRegressor()
            }

            logging.info('Evaluating models') 

            model_report=eval_model(Xtr,ytr,Xte,yte,models)
            print('\n')
            print('='*35)

            logging.info('Searching for best model')

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            print(f'Best model name: {best_model_name} , R2 score : {model_report[best_model_name]}')
            logging.info(f'Best model name: {best_model_name} , R2 score : {model_report[best_model_name]}')

            save_obj(self.model_trainer_config.trained_model_filepath,best_model)

        except Exception as e:
            logging.info('Error occured in initiating model training',e)
            