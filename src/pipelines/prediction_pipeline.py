from src.logger import logging
from src.exception import CustomException
import sys
import os
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            logging.info(f'Received features: {features}')
            print(f'Received Features: {features}')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts', 'model.pkl')
            logging.info('loading preprocessor')
            preprocessor=load_object(preprocessor_path)
            logging.info('loading model')
            model=load_object(model_path)
            logging.info('preprocessing data')
            data_scaled=preprocessor.transform(features)
            logging.info('predicting the data')
            pred=model.predict(data_scaled)
            logging.info('returning predicted value: ',pred)
            return pred

        except Exception as e:
            logging.info(f"Error in prediction pipeline {e}")

class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table,
        self.x=x,
        self.y=y,
        self.z=z,
        self.cut=cut,
        self.color=color,   
        self.clarity=clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_data={
                'carat':self.carat,
                'depth':self.depth,
                'table':self.table,
                'x':self.x,
                'y':self.y,
                'z':self.z,
                'cut':self.cut,
                'color':self.color,
                'clarity':self.clarity
            }

            df=pd.DataFrame(custom_data_input_data)
            # df=[[self.carat,self.depth,self.table,self.x,self.y,self.z,self.cut,self.color,self.clarity]]
            logging.info(f'The dataframe: \n{df.head().to_string()}')
            # logging.info(f'The list dataframe: {df}')
            logging.info('Dataframe Ready')
            return df


        except Exception as e:
            logging.info('Error occured in converting features to dataframe',e)
            raise CustomException(e,sys)