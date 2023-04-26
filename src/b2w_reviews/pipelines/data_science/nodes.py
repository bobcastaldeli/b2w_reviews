"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.6
"""


import logging
from typing import Any, Dict, Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split


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
