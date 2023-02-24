"""
This script contains nodes containing functions for pipeline
"""
import yaml
import datasets
import pandas as pd


with open('conf/base/parameters.yml', 'r') as f:
    params = yaml.safe_load(f)['parameters']


def download_reviews():
    """
    Load the reviews dataset from the datasets library. 
    and save in a csv file in data folder
    
    Args:
        params: parameters from parameters.yml file
    
    Returns:
        reviews: B2W reviews dataset from huggingface datasets library
    """
    dataset_name = params['dataset']['name']
    dataset_split = params['dataset']['split']
    dataset_path = params['dataset']['path']

    # load the dataset
    reviews = datasets.load_dataset(dataset_name)
    return reviews[dataset_split].to_csv(dataset_path, index=False)

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
