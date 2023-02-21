"""
This module contains the pipeline for the data engineering process.
"""


from kedro.pipeline import Pipeline, node
from .nodes import (
    process_data,
    split_data,
    train_model,
    evaluate_model,
    save_model,
    save_metrics,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=process_data,
                inputs="reviews",
                outputs="processed_reviews",
                name="process_data",
            ),
            node(
                func=split_data,
                inputs="processed_reviews",
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="model",
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model",
            ),
            node(
                func=save_model,
                inputs=["model", "X_train"],
                name="save_model",
            ),
            node(
                func=save_metrics,
                inputs=["metrics"],
                name="save_metrics",
            ),
        ]
    )
