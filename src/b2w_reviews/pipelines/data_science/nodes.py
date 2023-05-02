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