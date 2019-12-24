"""
This file contains the following methods:

replace_missing_new_category
replace_missing_remove_row
"""

import numpy as np
import pandas as pd
from aethos.util import _get_columns


def replace_missing_new_category(
    x_train, x_test=None, col_to_category=None, constant=None
):
    """
    Replaces missing values in categorical column with its own category. The categories can be autochosen
    from the defaults set.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset
        
    x_test : DataFrame
        Testing Dataset, by default None
        
    col_to_category : list or dict, optional
        A dictionary mapping column name to the category name you want to replace , by default None

    constant : str, int or float, optional
        Category placeholder value for missing values, by default None
    
    Returns
    -------
    Dataframe, *Dataframe:
        Cleaned columns of the Dataframe(s) provides with the provided constant.
        
    Returns 2 Dataframes if x_test is provided.

    Examples
    --------
    >>> ReplaceMissingCategory({'a': "Green", 'b': "Canada", 'c': "December"})
    >>> ReplaceMissingCategory("Blue", ['a', 'b', 'c'])
    """

    if isinstance(col_to_category, list):
        col_to_category = _get_columns(col_to_category, x_train)

    str_missing_categories = ["Other", "Unknown", "Missingx_trainCategory"]
    num_missing_categories = [-1, -999, -9999]

    if isinstance(col_to_category, dict):

        for col in col_to_category.keys():
            x_train[col].fillna(col_to_category[col], inplace=True)

            if x_test is not None:
                x_test[col].fillna(col_to_category[col], inplace=True)

    elif isinstance(col_to_category, list) and constant is not None:

        for col in col_to_category:
            x_train[col].fillna(constant, inplace=True)

            if x_test is not None:
                x_test[col].fillna(constant, inplace=True)

    else:

        for col in col_to_category:
            # Check if column is a number
            if np.issubdtype(x_train[col].dtype, np.number):
                new_category_name = _determine_default_category(
                    x_train, col, num_missing_categories
                )
                x_train[col].fillna(new_category_name, inplace=True)

                # Convert numeric categorical column to integer
                x_train[col] = x_train[col].astype(int)

                if x_test is not None:
                    x_test[col].fillna(new_category_name, inplace=True)
                    # Convert numeric categorical column to integer
                    x_test[col] = x_test[col].astype(int)
            else:
                new_category_name = _determine_default_category(
                    x_train, col, str_missing_categories
                )
                x_train[col].fillna(new_category_name, inplace=True)

                if x_test is not None:
                    new_category_name = _determine_default_category(
                        x_train, col, str_missing_categories
                    )
                    x_test[col].fillna(new_category_name, inplace=True)

    return x_train, x_test


def replace_missing_remove_row(x_train, x_test=None, cols_to_remove=[]):
    """
    Remove rows where the value of a column for those rows is missing.
        
    Parameters
    ----------
    x_train : DataFrame
        Dataset
        
    x_test : DataFrame
        Testing Dataset, by default None

    cols_to_remove : list
        List of columns you want to check to see if they have missing values in a row

    Returns
    -------
    Dataframe, *Dataframe:
        Cleaned columns of the Dataframe(s) provides with the provided constant.
        
    Returns 2 Dataframes if x_test is provided.
    """

    x_train = x_train.dropna(axis=0, subset=cols_to_remove)

    if x_test is not None:
        x_test = x_test.dropna(axis=0, subset=cols_to_remove)

    return x_train, x_test


def _determine_default_category(x_train, col, replacement_categories):
    """
    A utility function to help determine the default category name for a column that has missing
    categorical values. 
    
    It takes in a list of possible values and if any the first value in the list
    that is not a value in the column is the category that will be used to replace missing values.
    """

    unique_vals_col = x_train[col].unique()
    for potential_category in replacement_categories:

        # If the potential category is not already a category, it becomes the default missing category
        if potential_category not in unique_vals_col:
            new_category_name = potential_category
            break

    return new_category_name
