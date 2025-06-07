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
                        "KNN Regressor" : KNeighborsRegressor(),
                        "Decision Tree" : DecisionTreeRegressor(),
                        "Random Forest Regressor" : RandomForestRegressor(),
                        "XGB Regressor" : XGBRegressor(),
                        "Catboosting Regressor" : CatBoostRegressor(verbose=0),
                        "Adaboost Regressor" : AdaBoostRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor()

                    }
                    
                    params={
                        "Decision Tree": {
                             'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                             'splitter':['best', 'random'],
                             'max_features':['sqrt', 'log2']
                        },

                        "Random Forest Regressor":{
                             'n_estimators':[8,16,32,64,128,256],
                             'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                             'max_features':['sqrt', 'log2', None]
                        },

                        "Gradient Boosting": {
                             'learning_rate':[.1, .01 , .05, .001],
                             'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                             'n_estimators':[8,16,32,64,128,256]
                        },

                        "linear Regression": {},

                        "KNN Regressor":{
                             'n_neighbors': [5,7,9,11],
                             'weights':['uniform', 'distance'],
                             'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
                        },

                        "Catboosting Regressor": {
                            'iterations': [100, 200, 300],
                            'learning_rate': [0.01, 0.05, 0.1],
                            'depth': [4, 6, 8],
                            'l2_leaf_reg': [1, 3, 5]
                        },

                        "Adaboost Regressor": {
                             'n_estimators' :[8,16,32,64,128,256],
                             'loss':['linear', 'square', 'exponential'],
                             'learning_rate':[1,2,3,4,5]
                        },

                        "XGB Regressor":{
                             'learning_rate':[.1,.01,.05,.001],
                             'n_estimators':[8,16,32,64,128,256]
                        }
                    
                    }

                    model_report:dict=evaluate_models(x_train=x_train, y_train=y_train,x_test=x_test,y_test=y_test, models=models, param=params)

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
                    
