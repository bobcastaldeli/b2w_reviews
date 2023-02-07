"""
This script contains nodes containing functions for pipeline
"""

import datasets


def load_reviews_dataset(dataset: str):
    """
    Load the reviews dataset from the datasets library. 
    and save in a csv file in data folder
    
    Args:
        dataset: dataset to be loaded
    
    Returns:
        reviews: B2W reviews dataset from huggingface datasets library
    """
    # load the dataset
    reviews = datasets.load_dataset(dataset)
    return reviews['train'].to_csv('../data/01_raw/reviews.csv', index=False)
