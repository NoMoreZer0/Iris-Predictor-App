import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))

pytest.importorskip("streamlit")

from classification import load_data


def test_load_data_shape():
    df, target_names = load_data()

    assert df.shape == (150, 5)
    assert list(df.columns) == [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species",
    ]
    assert len(target_names) == 3


def test_random_forest_trained_on_loaded_data_is_accurate():
    df, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], df["species"], test_size=0.2, random_state=42
        )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    assert accuracy >= 0.9

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample)
    assert prediction.shape == (1,)


def test_logistic_regression_handles_loaded_data():
    df, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], df["species"], test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    assert accuracy >= 0.85
