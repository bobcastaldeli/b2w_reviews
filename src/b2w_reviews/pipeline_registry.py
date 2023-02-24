"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data_engineering.pipeline import create_pipeline


def register_pipelines():
    return {"__default__": create_pipeline()}