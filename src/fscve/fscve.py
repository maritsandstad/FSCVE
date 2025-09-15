"""
FSCVE the Forest Sensitive Climate Variable Emulator
"""

import numpy as np
import pandas as pd

from .data_readying_library import (
    check_data,
    cut_data_points_where_all_equal,
    fill_zeros,
)


class FSCVE:
    """
    FSCVE the Forest Sensitive Climate Variable Emulator

    Attributes
    ----------
    model : sklearn.model
        Regression model for the emulator. Can be pre-trained, and or trained
        and retrained from within the interface, or just used to make predictions
    predictor_list : list
        list of variable names for predictors to the Regression model
    variable_list : list
        list of variable names for variables that the Regression model can fit
    """

    def __init__(self, model, predictor_list, variable_list):
        """
        Initialize FSCVE instance

        Parameters
        ----------
        model : sklearn.model
            Regression model for the emulator. Can be pre-trained, and or trained
            and retrained from within the interface, or just used to make predictions
        predictor_list : list
            list of variable names for predictors to the Regression model
        variable_list : list
            list of variable names for variables that the Regression model can fit
        """
        self.model = model
        self.predictor_list = predictor_list
        self.variable_list = variable_list

    def predict_and_get_variable_diff(self, data_base, data_forest_change, fill_val=0):
        """
        Predict and get the diff in prediction between base and forest_change

        Parameters
        ----------
        data_base : pd.DataFrame
            Data presenting the base run forest configuration
        data_forest_change : pd.DataFrame
            Data presenting the run with forest change configuration
        fill_val : float
            Value to fill in for missing data. Default is 0, but np.nan can also
            be a useful option

        Returns
        -------
        pd.DataFrame
            Results of the difference in prediction due to the forest change
        """
        data_base = check_data(self.predictor_list, data_base)
        data_forest_change = check_data(self.predictor_list, data_forest_change)
        data_base_short, data_forest_short = cut_data_points_where_all_equal(
            data_base, data_forest_change
        )
        base_prediction = self._predict_from_variables(data_base_short)
        forest_prediction = self._predict_from_variables(data_forest_short)
        result = forest_prediction - base_prediction
        return fill_zeros(result, data_base, fill_val=fill_val)

    def _predict_from_variables(self, data):
        """
        Get prediction for dataset

        Parameters
        ----------
        data: pd.DataFrame
            Data for which to make predictions

        Returns
        -------
        pd.DataFrame
            Containing the prediction made for the data
        """
        prediction = self.model.predict_with_current(data)
        if isinstance(prediction, np.ndarray):
            prediction = pd.DataFrame(
                data=prediction, columns=self.variable_list, index=data.index
            )
        return prediction

    def retrain_model(self, Xtrain, ytrain):
        """
        Retrain the model with new training data

        Parameters
        ----------
        Xtrain : pd.DataFrame
            Training data which will be used to refit the model
        ytrain : np.ndarray
            Training data `truth` for the model to fit its predictions to
        """
        self.model.train_new_model_instance(Xtrain, ytrain, update_current=True)

    def evaluate_model(self, Xtest, ytest):
        """
        Evaluate the model fittness on test data

        Parameters
        ----------
        Xtest : pd.DataFrame
            Test data on which to evaluate the model fitness
        ytest : np.ndarray
            Test data `truth` to compare the test prediction to

        Returns
        -------
        list
            Containing first the vector of test predictions, followed by
            the evaluation metrics Mean Absolute Error, Mean Squared Error,
            Root Mean Squared Error and the R2 score in that order
        """
        ytest_pred, mae, mse, rmse, r2 = self.model.evaluate_model(Xtest, ytest)
        return ytest_pred, mae, mse, rmse, r2
