"""
Machine learning infrastructure
"""

import logging

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


def scale_x(Xtrain):
    """
    Make a scaler for training data and apply it to that dataset

    Parameters
    ----------
    Xtrain : pd.DataFrame
        Training data

    Returns
    -------
    list
        First item is scaler defined on the training dataset.
        Second is the transformed version of the training data
    """
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    return scaler, scaler.transform(Xtrain)


def setup_and_make_ready_dataset(X, y, scale_y=True):
    """
    Split data in training and test set, scale and returned transformed data

    Parameters
    ----------
    X : pd.DataFrame
        Predictor data
    y : pd.DataFrame
        Target data
    scale_y : bool
        Whether to apply scaling to target data

    Returns
    -------
    list
        First item is scaled predictor training data.
        Second item is the scaled predictor test data.
        Third item is the target training data (possibly scaled)
        Fourth item is the target test data (possibly scaled)
    """
    Xtrain, Xtest, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=100
    )
    scaler, Xtrain = scale_x(Xtrain)
    # Also scale targets?
    if scale_y:
        scalery = StandardScaler()
        scalery.fit(y_train)
        # print(y_test)
        y_train.replace(
            to_replace=y_train.values, value=scalery.transform(y_train), inplace=True
        )
        y_test.replace(
            to_replace=y_test.values, value=scalery.transform(y_test), inplace=True
        )
    # print(y_test)
    # sys.exit(4)
    # apply same transformation to test data
    Xtest = scaler.transform(Xtest.values)

    return Xtrain, Xtest, y_train, y_test


class UntrainedModelError(Exception):
    """
    Specialised error to throw when an Untrained Model is called on to make predictions
    """

    # Constructor or Initializer
    def __init__(self, value):  # pylint: disable=super-init-not-called
        self.value = value

    # __str__ is to print() the value
    def __str__(self):
        """
        Return the error value string
        """
        return repr(self.value)


class MLMODELINTERFACE:
    """
    Machine learning interface class

    Attributes
    ----------
    model_name : str
        Machine learning regressor name
    current_regressor: sklearn.model
        Intialised and trained specific instance of regressor
    scaler: sklearn.preprocessing.StandardScaler
        Scaler applied in the training for current instance  of regressor
    """

    def __init__(self, model):
        """
        Intialize MLMODELINTERFACE

        Parameters
        ----------
        model_name : str
            Machine learning regressor name
        """
        self.model_name = model
        self.current_regressor = None
        self.scaler = None

    def train_new_model_instance(
        self, Xtrain, ytrain, options=None, update_current=True, scale_x_dat=True
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        Train new model instance for regressor

        Parameters
        ----------
        Xtrain : pd.dataFrame
            Predictor training data to fit new regressor instance
        ytrain : pd.dataFrame
            Target training data to fit new regressor instance
        options: dict
            Dictionary listing options to send to the initialization
            of the new regressor instance
        update_current : bool
            Whether to update self.current_regressor to the results of
            this training, otherwise the resulting regressor will just
            be returned
        scale_x_dat : bool
            Whether to scale the training data before fitting, if update_current
            is True, this will also lead to updating self.scaler

        Returns
        -------
        sklearn.model, list
            If scale_x_dat is True a list with the new scaler and the regressor will be
            sent, otherwise only the regressor will be sent
        """
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
            return scaler, regressor
        return regressor

    def predict_with_current(self, Xtest):
        """
        Use self.current_regressor to make predictions

        Parameters
        ----------
        Xtest : pd.DataFrame
            Test predictor data

        Returns
        -------
        np.ndarray
            Predictions from the current regressor

        Raises
        ------
        UntrainedModelError
            If the model has not been trained yet
        """
        if self.current_regressor is None:
            LOGGER.error(  # pylint: disable=logging-fstring-interpolation
                f"Model {self.model_name} has not been trained yet, no predictions can be made"
            )
            raise UntrainedModelError(
                f"Model {self.model_name} has not been trained yet. Hence no prediction can be made"
            )
        if self.scaler:
            Xtest = self.scaler.transform(Xtest)
        return self.current_regressor.predict(Xtest)

    def evaluate_model(self, Xtest, ytest):
        """
        Use self.current_regressor to make predictions

        Parameters
        ----------
        Xtest : pd.DataFrame
            Test predictor data
        ytest : np.ndarray
            Test target `truth`

        Returns
        -------
        list
            Containing first the vector of test predictions, followed by
            the evaluation metrics Mean Absolute Error, Mean Squared Error,
            Root Mean Squared Error and the R2 score in that order

        Raises
        ------
            UntrainedModelError
            If the model has not been trained yet
        """
        ytest_pred = self.predict_with_current(Xtest)
        mae = metrics.mean_absolute_error(ytest, ytest_pred)
        mse = metrics.mean_squared_error(ytest, ytest_pred)
        rmse = np.sqrt(metrics.mean_squared_error(ytest, ytest_pred))
        r2 = metrics.r2_score(ytest, ytest_pred)
        return ytest_pred, mae, mse, rmse, r2
