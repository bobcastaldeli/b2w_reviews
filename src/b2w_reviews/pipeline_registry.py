"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data_engineering.pipeline import create_pipeline as de
from .pipelines.data_science.pipeline import create_pipeline as ds


def register_pipelines():
    data_engineering_pipeline = de()
    data_science_pipeline = ds()
    return {
        "de": data_engineering_pipeline,
        "ds": data_science_pipeline,
        "__default__": data_engineering_pipeline + data_science_pipeline,
    }