import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.logger import logging 
from src.exception import CustomException

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as f:
            pickle.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    
def eval_model(Xtr,ytr,Xte,yte,models):
    model_report={}
    for i in range(len(models)):
        model=list(models.values())[i]
        model.fit(Xtr,ytr)
        y_pred=model.predict(Xte)
        r2=r2_score(y_pred,yte)
        model_report[list(models.keys())[i]]=r2
        
    return model_report

def load_object(file_path):
    try:
        
        with open(file_path,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.info('Error occured in loading pickle',e)
        raise CustomException(e,sys)