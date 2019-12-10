"""
This file contains the following methods:

remove_columns_threshold
remove_rows_threshold
remove_duplicate_rows
remove_duplicate_columns
replace_missing_random_discrete
replace_missing_knn
replace_missing_interpolate
replace_missing_fill
replace_missing_indicator
remove_unique_columns_threshold
"""
import warnings

import numpy as np
import pandas as pd
from aethos.util import drop_replace_columns
from sklearn.impute import MissingIndicator, KNNImputer


def remove_columns_threshold(x_train, x_test=None, threshold=0.5):
    """
    Remove columns from the dataframe that have more than the threshold value of missing rows.
    Example: Remove columns where > 50% of the data is missing

    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.

    threshold : float
        Threshold value between 0 and 1 that if the column
        has more than the specified threshold of missing values, it is removed. 
        by default, 0.5
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    Returns 2 Dataframes if x_test is provided.
    """

    if threshold > 1 or threshold < 0:
        raise ValueError("Threshold cannot be greater than 1 or less than 0.")

    criteria_meeting_columns = x_train.columns[x_train.isnull().mean() < threshold]

    if x_test is not None:
        return x_train[criteria_meeting_columns], x_test[criteria_meeting_columns]
    else:
        return x_train[criteria_meeting_columns], None


def remove_rows_threshold(x_train: pd.DataFrame, x_test=None, threshold=0.5):
    """
    Remove rows from the dataframe that have more than the threshold value of missing rows.
    Example: Remove rows where > 50% of the data is missing
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Training dataset, by default None.

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.

    threshold : float
        Threshold value between 0 and 1 that if the row
        has more than the specified threshold of missing values, it is removed.
        by default, 0.5
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    if threshold > 1 or threshold < 0:
        raise ValueError("Threshold cannot be greater than 1 or less than 0.")

    x_train = x_train.dropna(thresh=round(x_train.shape[1] * threshold), axis=0)

    if x_test is not None:
        x_test = x_test.dropna(thresh=round(x_test.shape[1] * threshold), axis=0)

    return x_train, x_test


def remove_duplicate_rows(x_train, x_test=None, list_of_cols=[]):
    """
    Removes rows that are exact duplicates of each other.
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    list_of_cols: list, optional
        Columns to check if their values are duplicated, by default []

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    x_train = x_train.drop_duplicates(list_of_cols)

    if x_test is not None:
        x_test = x_test.drop_duplicates(list_of_cols)

    return x_train, x_test


def remove_duplicate_columns(x_train, x_test=None):
    """
    Removes columns whose values are exact duplicates of each other.
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset.

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    x_train = x_train.T.drop_duplicates().T

    if x_test is not None:
        x_test = x_test.T.drop_duplicates().T

    return x_train, x_test


def replace_missing_random_discrete(x_train, x_test=None, list_of_cols=[]):
    """
    Replace missing values in with a random number based off the distribution (number of occurences) 
    of the data.

    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
        
    list_of_cols : list
        A list of specific columns to apply this technique to, by default []   
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    for col in list_of_cols:
        probabilities = x_train[col].value_counts(normalize=True)

        missing_data = x_train[col].isnull()
        x_train.loc[missing_data, col] = np.random.choice(
            probabilities.index,
            size=len(x_train[missing_data]),
            replace=True,
            p=probabilities.values,
        )

        if x_test is not None:
            missing_data = x_test[col].isnull()
            x_test.loc[missing_data, col] = np.random.choice(
                probabilities.index,
                size=len(x_test[missing_data]),
                replace=True,
                p=probabilities.values,
            )

    return x_train, x_test


def replace_missing_knn(x_train, x_test=None, **knn_kwargs):
    """
    Replaces missing data using K nearest neighbors.
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.

    n_neighbors : int, optional
        Number of rows around the missing data to look at, by default 5
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    neighbors = knn_kwargs.pop("n_neighbors", 5)
    columns = x_train.columns
    knn = KNNImputer(n_neighbors=neighbors, **knn_kwargs)

    train_knn_transformed = knn.fit_transform(x_train.values)

    if x_test is not None:
        warnings.warn(
            "If your test data does not come from the same distribution of the training data, it may lead to erroneous results."
        )
        test_knn_transformed = knn.fit_transform(x_test.values)

    return (
        pd.DataFrame(data=train_knn_transformed, columns=columns),
        pd.DataFrame(data=test_knn_transformed, columns=columns),
    )


def replace_missing_interpolate(x_train, x_test=None, list_of_cols=[], **inter_kwargs):
    """
    Replaces missing data using interpolation techniques.
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
        
    list_of_cols : list
        A list of specific columns to apply this technique to, by default []
    
    method : str, optional
        Interpolation method, by default linear

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    method = inter_kwargs.pop("method", "linear")

    for col in list_of_cols:
        x_train[col] = x_train[col].interpolate(method=method, **inter_kwargs)

        if x_test is not None:
            warnings.warn(
                "If test data does not come from the same distribution of the training data, it may lead to erroneous results."
            )
            x_test[col] = x_test[col].interpolate(method=method, **inter_kwargs)

    return x_train, x_test


def replace_missing_fill(
    x_train, x_test=None, list_of_cols=[], method="", **extra_kwargs
):
    """
    Replaces missing values with the known values ahead of it and behind it.
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
        
    list_of_cols : list
        A list of specific columns to apply this technique to, by default []

    method : str
        Type of fill, by default ''
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    # Handle erroneous input
    extra_kwargs.pop("method", method)

    for col in list_of_cols:
        x_train[col] = x_train[col].fillna(method=method, **extra_kwargs)

        if x_test is not None:
            x_test[col] = x_test[col].fillna(method=method, **extra_kwargs)

    return x_train, x_test


def replace_missing_indicator(
    x_train,
    x_test=None,
    list_of_cols=[],
    missing_indicator=1,
    valid_indicator=0,
    keep_col=False,
):
    """
    Adds a new column describing if the column provided is missing data
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.

    list_of_cols : list
        A list of specific columns to apply this technique to, by default []

    missing_indicator : int, optional
        Value to indicate missing data, by default 1

    valid_indicator : int, optional
        Value to indicate non missing data, by default 0

    keep_col : bool, optional
        True to keep column, False to replace it, by default False
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    for col in list_of_cols:
        x_train[col + "_missing"] = [
            missing_indicator if x else valid_indicator for x in x_train[col].isnull()
        ]

        if not keep_col:
            x_train = x_train.drop([col], axis=1)

        if x_test is not None:
            x_test[col + "_missing"] = [
                missing_indicator if x else valid_indicator
                for x in x_test[col].isnull()
            ]

            if not keep_col:
                x_test = x_test.drop([col], axis=1)

    return x_train, x_test


def remove_constant_columns(x_train, x_test=None):
    """
    Removes columns that have only one unique value.

    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    # If the number of unique values is not 0(all missing) or 1(constant or constant + missing)
    keep_columns = list(
        filter(lambda x: x_train.nunique()[x] not in [0, 1], x_train.columns)
    )

    x_train = x_train[keep_columns]

    if x_test is not None:
        x_test = x_test[keep_columns]

    return x_train, x_test


def remove_unique_columns(x_train, x_test):
    """
    Remove columns that have only unique values.

    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    # If the number of unique values is not 0(all missing) or 1(constant or constant + missing)
    keep_columns = list(
        filter(lambda x: x_train.nunique()[x] != x_train.shape[0], x_train.columns)
    )

    x_train = x_train[keep_columns]

    if x_test is not None:
        x_test = x_test[keep_columns]

    return x_train, x_test
