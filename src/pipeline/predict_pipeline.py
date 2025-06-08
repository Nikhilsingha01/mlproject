import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class predictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path ='artifacts\model.pkl'
            preprecessor_path = 'artifacts\preprocessor.pkl'
            model =load_object(file_path=model_path)
            preprocessor =load_object(file_path=preprecessor_path)

            if features.isnull().values.any():
                print("Warning: Missing values found in input. Filling with default values.")
                # Fill or handle missing values (example strategy)
                features.fillna({
                    "gender": "male",  # or whatever was in training
                    "race/ethnicity": "group A",
                    "parental level of education": "high school",
                    "lunch": "standard",
                    "test preparation course": "none",
                    "reading score": 70,
                    "writing score": 70
                }, inplace=True)

            data_scaled = preprocessor.transform(features)
            preds= model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__( self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education ,
        lunch: str,
        test_preparation_course: str,
        reading_score : int,
        writing_score : int  ):
    
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender" : [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)

    
            