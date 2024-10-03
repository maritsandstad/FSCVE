import pandas as pd
import numpy as np
import logging

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  

LOGGER = logging.getLogger(__name__)

def scale_x(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler, scaler.transform(X_train)

def setup_and_make_ready_dataset(X, y, scale_y = True):
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    scaler, X_train = scale_x(X_train)
    # Also scale targets?
    if scale_y: 
        scalery = StandardScaler()
        scalery.fit(y_train)
        #print(y_test)
        y_train.replace(to_replace=y_train.values, value=scalery.transform(y_train), inplace=True)
        y_test.replace(to_replace=y_test.values, value=scalery.transform(y_test), inplace=True)
    #print(y_test)
    #sys.exit(4)
    # apply same transformation to test data
    X_test = scaler.transform(X_test.values)

    return X_train,X_test, y_train,y_test

class UntrainedModelError(Exception):
 
    # Constructor or Initializer
    def __init__(self, value):
        self.value = value
 
    # __str__ is to print() the value
    def __str__(self):
        return(repr(self.value))

class MLMODELINTERFACE: 
    # TODO: keep track of scaler object to be used...
    def __init__(self, model):
        self.model_name = model
        self.current_regressor = None
        self.scaler = None

    def train_new_model_instance(self, Xtrain, ytrain, options= None, update_current= True, scale_x_dat = True):
        if options: 
            regressor = self.model_name(**options)
        else: 
            regressor = self.model_name()

        if scale_x_dat:
            scaler, Xtrain = scale_x(Xtrain)

        regressor.fit(Xtrain, ytrain)

        if update_current:
            self.current_regressor = regressor
            if scale_x_dat:
                self.scaler = scaler
        if scale_x_dat:
            return scaler,regressor
        return regressor
    
    def predict_with_current(self, Xtest):
        if self.current_regressor is None:
            LOGGER.error(f"Model {self.model_name} has not been trained yet, no predictions can be made")
            raise UntrainedModelError(
                f"Model {self.model_name} has not been trained yet. Hence no prediction can be made"    
        )
        if self.scaler:
            Xtest = self.scaler.transform(Xtest)
        return self.current_regressor.predict(Xtest)

    def evaluate_model(self, Xtest, ytest):
        ytest_pred = self.predict_with_current(Xtest)
        mae = metrics.mean_absolute_error(ytest, ytest_pred)
        mse = metrics.mean_squared_error(ytest, ytest_pred)
        rmse = np.sqrt(metrics.mean_squared_error(ytest, ytest_pred))
        r2 = metrics.r2_score(ytest, ytest_pred)
        return ytest_pred, mae, mse, rmse, r2   

