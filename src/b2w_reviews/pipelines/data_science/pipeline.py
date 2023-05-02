"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=['reviews_features', 'parameters'],
                outputs=['train_set', 'test_set'],
                name='split_data',
            ),
            node(
                func=train_model,
                inputs=['train_set', 'parameters'],
                outputs=['model', 'labels'],
                name='train_model',
            )
        ],
    )
