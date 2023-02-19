"""
This script contains nodes containing functions for pipeline
"""

import datasets
import pandas as pd


def load_reviews_dataset(dataset: str, path: str):
    """
    Load the reviews dataset from the datasets library. 
    and save in a csv file in data folder
    
    Args:
        dataset: dataset to be loaded
        path: path to the reviews dataset
    
    Returns:
        reviews: B2W reviews dataset from huggingface datasets library
    """
    # load the dataset
    reviews = datasets.load_dataset(dataset)
    return reviews['train'].to_csv(path, index=False)


def read_reviews_dataset(path: str):
    """
    Read the reviews dataset from the data folder
    
    Args:
        path: path to the reviews dataset
    
    Returns:
        reviews: B2W reviews dataset
    """
    # read the dataset
    reviews = pd.read_csv(path)
    return reviews


def drop_null_values(dataframe: pd.DataFrame, column: str):
    """
    Drop all the rows with null values in a column
    
    Args:
        dataframe: dataframe to be cleaned
        column: column to be checked
    
    Returns:
        dataframe: cleaned dataframe
    """
    dataframe = dataframe.dropna(subset=[column])
    return dataframe


def save_dataframe(dataframe: pd.DataFrame, path: str):
    """
    Save the dataframe in a csv file
    
    Args:
        dataframe: dataframe to be saved
        path: path to save the dataframe
    """
    return dataframe.to_csv(path, index=False)
