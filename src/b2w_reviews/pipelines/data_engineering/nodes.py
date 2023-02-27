"""
This script contains nodes containing functions for pipeline
"""
import yaml
from typing import Any, Callable, Dict, Tuple
import datasets
import pandas as pd
from sklearn.model_selection import train_test_split


#with open('conf/base/parameters.yml', 'r') as f:
#    params = yaml.safe_load(f)['parameters']


def download_reviews(parameters: Dict[str, Any]):
    """
    Load the reviews dataset from the datasets library. 
    and save in a csv file in data folder
    
    Args:
        parameters: parameters from parameters.yml file
    
    Returns:
        reviews: B2W reviews dataset from huggingface datasets library
    """  

    # load the dataset
    dataset = datasets.load_dataset(parameters['name'])
    return dataset[parameters['split']].to_pandas()



def drop_null_values(dataframe: pd.DataFrame, parameters: list) -> pd.DataFrame:
    """
    Drop null values from the dataframe
    
    Args:
        dataframe: dataframe to be cleaned
        parameters: parameters from parameters.yml file
    
    Returns:
        dataframe: cleaned dataframe
    """
    print(f"Input data columns: {dataframe.columns}")
    print(f"Columns to drop nulls from: {parameters['columns']}")
    cleaned_data = dataframe.dropna(subset=parameters['columns'])
    return cleaned_data


def split_data(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into train and test
    
    Args:
        dataframe: dataframe to be split
    
    Returns:
        train: train dataframe
        test: test dataframe
    """
    X = pd.DataFrame(dataframe['review_text'])
    y = pd.DataFrame(dataframe['overall_rating'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test