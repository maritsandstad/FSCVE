"""
FSCVE the Forest Sensitive Climate Variable Emulator
"""
import numpy as np
import pandas as pd

from .data_readying_library import check_data, cut_data_points_where_all_equal, fill_zeros

class FSCVE:
   """
   FSCVE the Forest Sensitive Climate Variable Emulator
   """ 
   def __init__(self, model, predictor_list, varible_list):
      self.model = model
      self.predictor_list = predictor_list
      self.variable_list = varible_list


   def predict_and_get_variable_diff(self, data_base, data_forest_change):
       """
       """
       data_base = check_data(self.predictor_list, data_base)
       data_forest_change = check_data(self.predictor_list, data_forest_change)
       data_base_short, data_forest_short = cut_data_points_where_all_equal(data_base, data_forest_change)
       base_prediction = self._predict_from_variables(data_base_short)
       forest_prediction =self._predict_from_variables(data_forest_short)
       result = base_prediction - forest_prediction

       return fill_zeros(result, data_base)

   def _predict_from_variables(self, data):
       
       prediction = self.model.predict_with_current(data)
       if isinstance(prediction, np.ndarray):
           prediction = pd.DataFrame(data= prediction, columns=self.variable_list, index=data.index)    
       return prediction
    
   def retrain_model(self, Xtrain, ytrain):
       self.model.train_new_model_instance(Xtrain, ytrain, update_current=True)

   def evaluate_model(self, Xtest, ytest):
      ytest_pred, mae, mse, rmse, r2 = self.model.evaluate_model(Xtest, ytest)
      return ytest_pred, mae, mse, rmse, r2