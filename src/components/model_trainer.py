import os
import sys
from dataclasses import dataclass# Basic Import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# modellinng
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost.sklearn import  XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
        def __init__(self):
            self.model_trainer_config = ModelTrainerConfig()

        def initiate_model_trainer(self , train_array, test_array):
                try:
                    logging.info("split training and test input data")
                    x_train , y_train, x_test, y_test =(
                        train_array[:,:-1],
                        train_array[:,-1],
                        test_array[:,:-1],
                        test_array[:,-1],
                    )

                    models = {
                        "linear Regression": LinearRegression(),
                        "k-Neighbours Regressor" : KNeighborsRegressor(),
                        "Decision Tree" : DecisionTreeRegressor(),
                        "Random Forest Regressor" : RandomForestRegressor(),
                        "XGB Regressor" : XGBRegressor(),
                        "Catboosting Regressor" : CatBoostRegressor(verbose=0),
                        "Adaboost Regressor" : AdaBoostRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor()

                        }
                    
                    model_report:dict=evaluate_models(x_train=x_train, y_train=y_train,x_test=x_test,y_test=y_test, models=models)

                    ## to get best model score from dict
                    best_model_score = max(sorted(model_report.values()))

                    ## to get best model name from dict
                    best_model_name =list(model_report.keys())[
                        list(model_report.values()).index(best_model_score)
                    ]

                    best_model = models[best_model_name]

                    if best_model_score<0.6:
                        raise CustomException ("No best model found")
                    
                    logging.info("Best found model on both training and test data")

                    save_object(
                        file_path=self.model_trainer_config.trained_model_file_path,
                        obj= best_model
                    )

                    predicted= best_model.predict(x_test)
                    r2_square = r2_score(y_test, predicted)

                    return r2_square
                

                except Exception as e:
                    raise CustomException(e, sys)
                    
