import pytest
import pandas as pd
import numpy as np
from fscve import ml_modelling_infrastructure
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def test_mlmodelinterface():
    mlinterface = ml_modelling_infrastructure.MLMODELINTERFACE(LinearRegression)
    Xtrain = pd.DataFrame(data = [[59.9,10.7,0,0],[52.3,4.9,0,0], [38.0,23.7,0,0],[59.3,18.1,0,0]], columns = ["LAT", "LON", "agb", "conifer"], index = ["Oslo", "Amsterdam", "Athens", "Stockholm"])
    ytrain = [18, 16, 29, 18]
    with pytest.raises(ml_modelling_infrastructure.UntrainedModelError, match = f"Model {mlinterface.model_name} has not been trained yet. Hence no prediction can be made"):
        mlinterface.evaluate_model(Xtrain, ytrain)

    mlinterface.train_new_model_instance(Xtrain, ytrain, scale_x_dat = False)

    Xtest =  pd.DataFrame(data = [[51.5,-0.1,0,0],[48.9,2.3,0,0], [38.7,-9.1,0,0]], columns = ["LAT", "LON", "agb", "conifer"], index = ["Paris", "Lisbon", "London"])
    ytest = [20, 19, 24]
    ytest_predtr, maetr, msetr, rmsetr, r2tr = mlinterface.evaluate_model(Xtrain, ytrain)
    ytest_predts, maets, msets, rmsets, r2ts = mlinterface.evaluate_model(Xtest, ytest)
    assert len(ytest_predtr) == 4
    assert len(ytest_predts) == 3
    assert maetr < maets
    assert msetr < msets
    assert rmsetr < rmsets
    assert r2tr > r2ts
    scale_x, other_regressor = mlinterface.train_new_model_instance(Xtest, ytest, options = {"fit_intercept": True}, update_current=False)
    ytest_pred2 = other_regressor.predict(scale_x.transform(Xtest))
    assert np.sqrt(metrics.mean_squared_error(ytest_pred2, ytest)) < rmsets
