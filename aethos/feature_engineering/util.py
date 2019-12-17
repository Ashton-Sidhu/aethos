"""
This file contains the following functions:

pca
apply
"""

import pandas as pd
import swifter
from sklearn.decomposition import PCA


def pca(x_train, x_test=None, **pca_kwargs):
    """
    Performs Principal Component Analysis on a dataset.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    pca_kwargs : dict or key word args
        PCA properties
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column

    Returns 2 Dataframes if x_test is provided. 
    """

    pca = PCA(**pca_kwargs)

    x_train = pd.DataFrame(pca.fit_transform(x_train))

    if x_test is not None:
        x_test = pd.DataFrame(pca.transform(x_test))

    return x_train, x_test


def apply(x_train, func, output_col: str, x_test=None):
    """
    Wrapper for pandas apply function to be used in this library. Applies `func` to the entire data
    or just the trianing and testing data

    Parameters
    ----------
    x_train : DataFrame
        Dataset

    func : Function pointer
        Function describing the transformation for the new column

    output_col : str
        New column name
        
    x_test : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column

    Returns 2 Dataframes if x_test is provided. 
    """

    x_train.loc[:, output_col] = x_train.swifter.progress_bar().apply(func, axis=1)

    if x_test is not None:
        x_test.loc[:, output_col] = x_test.swifter.progress_bar().apply(func, axis=1)

    return x_train, x_test
