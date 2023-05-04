"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.6
"""


import logging
from typing import Any, Callable, Dict, Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)
from embetter.grab import ColumnGrabber
from embetter.text import SentenceEncoder


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def split_data(
    reviews_clean: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Node for splitting the data into train and test sets.
    Args:
        reviews_clean: The cleaned data.
        parameters: A dictionary of parameters.
    Returns:
        X_train: The training set features.
        X_test: The test set features.
        y_train: The training set targets.
        y_test: The test set targets.
    """
    train, test = train_test_split(
        reviews_clean[[parameters['textcolumn'], parameters['sentimentcolumn']]],
        test_size=parameters["testsize"],
        random_state=parameters["randomseed"],
    )
    return train, test


def train_model(
    train_set: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[Callable, Dict[str, Any]]:
    """Node for encodign label and training a model.
    Args:
        X_train: The training set features.
        y_train: The training set targets.
        parameters: A dictionary of parameters.
    Returns:
        pipeline: The pickle pipeline.
    """
    train_set[parameters["textcolumn"]] = train_set[parameters["textcolumn"]].astype(str)
    train_features = pd.DataFrame(train_set[parameters["textcolumn"]])
    y_train = pd.DataFrame(train_set[parameters["sentimentcolumn"]])
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    train_label = label_encoder.transform(y_train)
    pipeline = make_pipeline(
        ColumnGrabber(parameters["textcolumn"]),
        SentenceEncoder(parameters["sentencemodel"]),
        LogisticRegression(random_state=parameters["randomseed"]),
    )
    pipeline.fit(train_features, train_label)
    return pipeline, label_encoder


def predict_model(
    model: Tuple[Callable, Dict[str, Any]], test_set: pd.DataFrame, parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Node for predicting the test set.
    Args:
        model: The trained model.
        X_test: The test set features.
    Returns:
        y_pred: The predicted targets.
    """
    pipeline = model
    test_set[parameters["textcolumn"]] = test_set[parameters["textcolumn"]].astype(str)
    test_features = pd.DataFrame(test_set[parameters["textcolumn"]])
    y_pred = pd.DataFrame(pipeline.predict(test_features))
    return y_pred


def evaluate_model(
    test_set: pd.DataFrame, y_pred: pd.DataFrame, label: Tuple[Callable, Dict[str, Any]], parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Node for evaluating the model.
    Args:
        y_test: The test set targets.
        y_pred: The predicted targets.
        model: The trained model.
    Returns:
        scores: A dictionary of evaluation scores.
    """
    label_encoder = label
    y_test = pd.DataFrame(test_set[parameters["sentimentcolumn"]])
    y_test = label_encoder.transform(y_test)
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, y_pred, average="macro"),
    }
    # transform scores to a dataframe
    scores = pd.DataFrame(scores, index=[0])
    return scores
