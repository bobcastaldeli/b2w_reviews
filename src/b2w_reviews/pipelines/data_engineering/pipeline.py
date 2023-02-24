"""
This module contains the pipeline for the data engineering process.
"""
from kedro.pipeline import Pipeline, node
from .nodes import download_reviews


def create_pipeline(**kwargs):

    return Pipeline(
        [
            node(
                func=download_reviews,
                inputs=None,
                outputs="reviews.csv",
                name="download_reviews",
            ),
        ]
    )
