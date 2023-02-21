"""
This module contains the pipeline for the data engineering process.
"""


from kedro.pipeline import Pipeline, node
from .nodes import (
    load_reviews_dataset,
    read_reviews_dataset,
    drop_null_values,
    save_dataframe,
)


def create_pipeline(**kwargs):

    return Pipeline(
        [
            node(
                func=load_reviews_dataset,
                inputs=None,
                outputs="reviews",
                name="load_reviews_dataset",
                params={"dataset": "b2w_reviews", "path": "data/01_raw/reviews.csv"},
            ),
            node(
                func=read_reviews_dataset,
                inputs="reviews",
                outputs="reviews",
                name="read_reviews_dataset",
                params={"path": "data/01_raw/reviews.csv"},
            ),
            node(
                func=drop_null_values,
                inputs="reviews",
                outputs="reviews",
                name="drop_null_values",
                params={"column": "product_name"},
            ),
            node(
                func=save_dataframe,
                inputs="reviews",
                outputs=None,
                name="save_dataframe",
                params={"path": "data/02_intermediate/reviews.csv"},
            ),
        ]
    )
