"""
This module contains the pipeline for the data engineering process.
"""
from kedro.pipeline import Pipeline, pipeline, node
from .nodes import download_reviews, drop_null_values


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(
                func=download_reviews,
                inputs=["parameters"],
                outputs="reviews_raw",
                name="download_reviews",
            ),
            node(
                func=drop_null_values,
                inputs=["reviews_raw", "parameters"],
                outputs="reviews_cleaned",
                name="drop_null_values",
            ),
        ]
    )
