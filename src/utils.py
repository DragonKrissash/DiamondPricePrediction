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