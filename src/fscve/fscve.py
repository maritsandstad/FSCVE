"""
FSCVE the Forest Sensitive Climate Variable Emulator
"""

class FSCVE:
    """
    FSCVE the Forest Sensitive Climate Variable Emulator
    """ 
    def __init__(self, model, predictor_list, varible_list):
      self.model = model
      self.predictor_list
      self.variable_list


    def predict_and_get_variable_diff(self, data_base, data_forest_change):
       """
       """
       data_base_short, data_forest_short = cut_data_points_where_all_equal(data_base, data_forest_change)
       base_prediction = self._predict_from_variables(data_base_short)
       forest_prediction =self._predict_from_variables(data_forest_change_short)
       result = base_prediction - forest_prediction

       return fill_zeros(result, data_base, data_base_short)

    def _predict_from_variables(data):
       self._check_data(data)
       prediction = self.model