"""Regression tests for preserving feature names through preprocessing."""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from model_input import transform_preserving_columns


def test_imputed_random_forest_input_keeps_feature_names():
    train = pd.DataFrame(
        {
            "temperature": [60.0, np.nan, 75.0],
            "ridership": [100.0, 150.0, 200.0],
        }
    )
    imputer = SimpleImputer(strategy="median")
    train_imputed = pd.DataFrame(
        imputer.fit_transform(train),
        columns=train.columns,
        index=train.index,
    )
    model = RandomForestRegressor(n_estimators=2, random_state=0).fit(
        train_imputed,
        [1.0, 2.0, 3.0],
    )

    prediction_input = transform_preserving_columns(imputer, train.iloc[[0]])

    assert prediction_input.columns.tolist() == train.columns.tolist()
    assert prediction_input.index.tolist() == [0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(prediction_input)

    assert not any("valid feature names" in str(item.message) for item in caught)
