'''
This file contains the following methods:

replace_missing_new_category
replace_missing_remove_row
'''

import numpy as np
import pandas as pd
from pyautoml.util import _function_input_validation, _get_columns

#TODO: Implement KNN, and replacing with most common category 

def replace_missing_new_category(col_to_category=None, constant=None, **datasets):
    """
    Replaces missing values in categorical column with its own category. The categories can be autochosen
    from the defaults set.

    Either the full data or training data plus testing data MUST be provided, not both.
    
    Parameters
    ----------
    col_to_category : list or dict, optional
        A dictionary mapping column name to the category name you want to replace , by default None

    constant : str, int or float, optional
        Category placeholder value for missing values, by default None

    data : DataFrame
        Full dataset, by default None

    x_train : DataFrame
        Training dataset, by default None

    x_test : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe, *Dataframe:
        Cleaned columns of the dataframe(s) provides with the provided constant.
        
    Returns 2 Dataframes if Train and Test data is provided.

    Examples
    --------
    >>> ReplaceMissingCategory({'a': "Green", 'b': "Canada", 'c': "December"})
    
    >>> ReplaceMissingCategory("Blue", ['a', 'b', 'c'])
    """

    data = datasets.pop('data', None)
    x_train = datasets.pop('x_train', None)
    x_test = datasets.pop('x_test', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))    

    if not _function_input_validation(data, x_train, x_test):
        raise ValueError("Please provide a full data or training and testing data.")

    if isinstance(col_to_category, list):
        col_to_category = _get_columns(col_to_category, data, x_train)
    
    str_missing_categories = ["Other", "Unknown", "MissingDataCategory"]
    num_missing_categories = [-1, -999, -9999]

    if isinstance(col_to_category, dict):        
        if data is not None:
            for col in col_to_category.keys():
                data[col].fillna(col_to_category[col], inplace=True)

            return data

        else:
            for col in col_to_category.keys():
                x_train[col].fillna(col_to_category[col], inplace=True)
                x_test[col].fillna(col_to_category[col], inplace=True)
            
            return x_train, x_test

    elif isinstance(col_to_category, list) and constant is not None:
        if data is not None:
            for col in col_to_category:
                data[col].fillna(constant, inplace=True)

            return data

        else:
            for col in col_to_category:
                x_train[col].fillna(constant, inplace=True)
                x_test[col].fillna(constant, inplace=True)
            
            return x_train, x_test

    else:
        if data is not None:
            for col in col_to_category:
                #Check if column is a number
                if np.issubdtype(data[col].dtype, np.number):
                    new_category_name = _determine_default_category(data, col, num_missing_categories)
                    data[col].fillna(new_category_name, inplace=True)

                    #Convert numeric categorical column to integer
                    data[col] = data[col].astype(int)
                else:
                    new_category_name = _determine_default_category(data, col, str_missing_categories)
                    data[col].fillna(new_category_name, inplace=True)           

            return data
        
        else:
            for col in col_to_category:                
                #Check if column is a number
                if np.issubdtype(x_train[col].dtype, np.number):
                    new_category_name = _determine_default_category(x_train, col, num_missing_categories)
                    x_train[col].fillna(new_category_name, inplace=True)
                    x_test[col].fillna(new_category_name, inplace=True)
                    #Convert numeric categorical column to integer
                    x_train[col] = x_train[col].astype(int)
                    x_test[col] = x_test[col].astype(int)

                else:
                    new_category_name = _determine_default_category(x_train, col, str_missing_categories)
                    x_train[col].fillna(new_category_name, inplace=True)
                    x_test[col].fillna(new_category_name, inplace=True)           

            return x_train, x_test

def replace_missing_remove_row(cols_to_remove: list, **datasets):
    """
    Remove rows where the value of a column for those rows is missing.
    
    Either the full data or training data plus testing data MUST be provided, not both.
    
    Parameters
    ----------
    cols_to_remove : list
        List of columns you want to check to see if they have missing values in a row

    data : DataFrame
        Full dataset, by default None

    x_train : DataFrame
        Training dataset, by default None
        
    x_test : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe, *Dataframe:
        Cleaned columns of the dataframe(s) provides with the provided constant.
        
    Returns 2 Dataframes if Train and Test data is provided.        
    """

    data = datasets.pop('data', None)
    x_train = datasets.pop('x_train', None)
    x_test = datasets.pop('x_test', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))  

    if not _function_input_validation(data, x_train, x_test):
        raise ValueError("Please provide a full data or training and testing data.")

    if data is not None:
        data = data.dropna(axis=0, subset=cols_to_remove)

        return data
    else:
        x_train = x_train.dropna(axis=0, subset=cols_to_remove)
        x_test = x_test.dropna(axis=0, subset=cols_to_remove)

        return x_train, x_test

def _determine_default_category(data, col, replacement_categories):
    """
    A utility function to help determine the default category name for a column that has missing
    categorical values. 
    
    It takes in a list of possible values and if any the first value in the list
    that is not a value in the column is the category that will be used to replace missing values.
    """

    unique_vals_col = data[col].unique()
    for potential_category in replacement_categories:

        #If the potential category is not already a category, it becomes the default missing category 
        if potential_category not in unique_vals_col:
            new_category_name = potential_category
            break

    return new_category_name
