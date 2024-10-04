import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from fscve import FSCVE, ml_modelling_infrastructure


def test_fscve():
    Xtrain = pd.DataFrame(
        data=[
            [59.9, 10.7, 1, 0.5],
            [52.3, 4.9, 1, 0.5],
            [38.0, 23.7, 0, 0],
            [59.3, 18.1, 0, 0],
        ],
        columns=["LAT", "LON", "agb", "conifer"],
        index=["Oslo", "Amsterdam", "Athens", "Stockholm"],
    )
    ytrain = [18, 16, 29, 18]
    full_ds = Xtrain.copy()
    full_ds["July temp"] = ytrain
    ml_linear = ml_modelling_infrastructure.MLMODELINTERFACE(LinearRegression)
    ml_linear.train_new_model_instance(Xtrain, ytrain)

    fscve_instance = FSCVE(
        ml_linear,
        predictor_list=["LAT", "LON", "agb", "conifer"],
        varible_list=["July temp"],
    )
    Xforest_base = pd.DataFrame(
        data=[[51.5, -0.1, 0, 0], [48.9, 2.3, 0, 0], [38.7, -9.1, 0, 0]],
        columns=["LAT", "LON", "agb", "conifer"],
        index=["Paris", "Lisbon", "London"],
    )
    Xforest_diff = pd.DataFrame(
        data=[[51.5, -0.1, 1, 0], [48.9, 2.3, 0, 1], [38.7, -9.1, 0, 0]],
        columns=["LAT", "LON", "agb", "conifer"],
        index=["Paris", "Lisbon", "London"],
    )

    prediction = fscve_instance.predict_and_get_variable_diff(
        Xforest_base, Xforest_diff
    )
    print(prediction)
    assert prediction.shape == (3, 1)
    assert set(prediction.index) == set(["Paris", "London", "Lisbon"])
    assert prediction.loc["London"].values == 0.0

    ytest = [20, 19, 24]
    fscve_instance.retrain_model(Xtrain=Xforest_base, ytrain=ytest)
    ytest_pred1, mae1, mse1, rmse1, r21 = fscve_instance.evaluate_model(
        Xforest_base, ytest
    )
    ytest_pred2, mae2, mse2, rmse2, r22 = fscve_instance.evaluate_model(
        Xforest_diff, ytest
    )
    assert len(ytest_pred1) == len(ytest_pred2)
    assert mae1 <= mae2
    assert mse1 <= mse2
    assert rmse1 <= rmse2
    assert r21 >= r22
