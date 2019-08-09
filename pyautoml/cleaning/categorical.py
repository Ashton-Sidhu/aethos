'''
This file contains the following methods:

ReplaceMissingNewCategory
ReplaceMissingRemoveRow
'''

import numpy as np
import pandas as pd
from pyautoml.util import DropAndReplaceColumns, _FunctionInputValidation

#TODO: Implement KNN, and replacing with most common category 

def ReplaceMissingNewCategory(constant=None, col_to_category=None, **datasets):
    """
    Replaces missing values in categorical column with its own category. The categories can be autochosen
    from the defaults set.
    
    Args:
        col_to_category (list or dict, optional): A dictionary mapping column name to the category name you want to replace 
        missing values with. Defaults to None.
        constant (any, optional):  Category placeholder value for missing values. Defaults to None.

        Either the full data or training data plus testing data MUST be provided, not both.

        data {DataFrame} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})
        
    Returns:
        Dataframe, *Dataframe: Cleaned columns of the dataframe(s) provides with the provided constant.

        * Returns 2 Dataframes if Train and Test data is provided.

    Examples:

        >>>> ReplaceMissingCategory({'a': "Green", 'b': "Canada", 'c': "December"})
        >>>> ReplaceMissingCategory("Blue", ['a', 'b', 'c'])
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError(f"Invalid parameters passed: {str(datasets)}")    

    if not _FunctionInputValidation(data, train_data, test_data):
        raise ValueError("Please provide a full data or training and testing data.")
    
    str_missing_categories = ["Other", "Unknown", "MissingDataCategory"]
    num_missing_categories = [-1, -999, -9999]

    if isinstance(col_to_category, dict):
        
        if data is not None:
            for col in col_to_category.keys():

                data[col].fillna(col_to_category[col], inplace=True)

            return data

        else:

            for col in col_to_category.keys():

                train_data[col].fillna(col_to_category[col], inplace=True)
                test_data[col].fillna(col_to_category[col], inplace=True)
            
            return train_data, test_data


    elif isinstance(col_to_category, list) and constant is not None:

        if data is not None:
            for col in col_to_category:

                data[col].fillna(constant, inplace=True)

            return data

        else:
            for col in col_to_category:

                train_data[col].fillna(constant, inplace=True)
                test_data[col].fillna(constant, inplace=True)
            
            return train_data, test_data


    elif isinstance(col_to_category, list) and constant is None:

        if data is not None:
            for col in col_to_category:

                #Check if column is a number
                if np.issubdtype(data[col].dtype, np.number):
                    new_category_name = _DetermineDefaultCategory(data, col, num_missing_categories)
                    data[col].fillna(new_category_name, inplace=True)
                    #Convert numeric categorical column to integer
                    data[col] = data[col].astype(int)

                else:
                    new_category_name = _DetermineDefaultCategory(data, col, str_missing_categories)
                    data[col].fillna(new_category_name, inplace=True)           

            return data
        
        else:
            for col in col_to_category:
                
                #Check if column is a number
                if np.issubdtype(train_data[col].dtype, np.number):
                    new_category_name = _DetermineDefaultCategory(train_data, col, num_missing_categories)
                    train_data[col].fillna(new_category_name, inplace=True)
                    test_data[col].fillna(new_category_name, inplace=True)
                    #Convert numeric categorical column to integer
                    train_data[col] = train_data[col].astype(int)
                    test_data[col] = test_data[col].astype(int)

                else:
                    new_category_name = _DetermineDefaultCategory(train_data, col, str_missing_categories)
                    train_data[col].fillna(new_category_name, inplace=True)
                    test_data[col].fillna(new_category_name, inplace=True)           

            return train_data, test_data


    elif col_to_category is None and constant is not None:

        if data is not None:

            data = data.fillna(constant)

            return data
        
        else:

            train_data = train_data.fillna(constant)
            test_data = test_data.fillna(constant)

            return train_data, test_data        

    else:

        if data is not None:
            for col in data.columns:

                if np.issubdtype(data[col].dtype, np.number):
                    new_category_name = _DetermineDefaultCategory(data, col, num_missing_categories)
                    #Convert numeric categorical column to integer
                    data[col].fillna(new_category_name, inplace=True)
                    data[col] = data[col].astype(int)

                else:
                    new_category_name = _DetermineDefaultCategory(data, col, str_missing_categories)
                    data[col].fillna(new_category_name, inplace=True)                

            return data            
        
        else:

            for col in data.columns:

                #Check if column is a number
                if np.issubdtype(train_data[col].dtype, np.number):
                    new_category_name = _DetermineDefaultCategory(train_data, col, num_missing_categories)
                    train_data[col].fillna(new_category_name, inplace=True)
                    test_data[col].fillna(new_category_name, inplace=True)
                    #Convert numeric categorical column to integer
                    train_data[col] = train_data[col].astype(int)
                    test_data[col] = test_data[col].astype(int)

                else:
                    new_category_name = _DetermineDefaultCategory(train_data, col, str_missing_categories)
                    train_data[col].fillna(new_category_name, inplace=True)
                    test_data[col].fillna(new_category_name, inplace=True)

            return train_data, test_data   

def ReplaceMissingRemoveRow(cols_to_remove, **datasets):
    """
    Remove rows where the value of a column for those rows is missing.
    
    Args:
        cols_to_remove (list): List of columns you want to check to see if they have missing values in a row 
    
        Either the full data or training data plus testing data MUST be provided, not both.

        data {DataFrame} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})

    Returns:
        Dataframe, *Dataframe: Transformed dataframe with rows with a missing values in a specific column are missing

        * Returns 2 Dataframes if Train and Test data is provided.  
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError(f"Invalid parameters passed: {str(datasets)}")  

    if not _FunctionInputValidation(data, train_data, test_data):
        raise ValueError("Please provide a full data or training and testing data.")

    if data is not None:
        data = data.dropna(axis=0, subset=cols_to_remove)

        return data
    else:
        train_data = train_data.dropna(axis=0, subset=cols_to_remove)
        test_data = test_data.dropna(axis=0, subset=cols_to_remove)

        return train_data, test_data

def _DetermineDefaultCategory(data, col, replacement_categories):
    """A utility function to help determine the default category name for a column that has missing
    categorical values. It takes in a list of possible values and if any the first value in the list
    that is not a value in the column is the category that will be used to replace missing values.
    
    Arguments:
        col {string} -- Column of the dataframe
        replacement_categories {list} -- List of potential category names to replace missing values with
    
    Returns:
        [type of contents of replacement_categories (str, int, etc.)] -- the default category for that column 
    """

    unique_vals_col = data[col].unique()
    for potential_category in replacement_categories:

        #If the potential category is not already a category, it becomes the default missing category 
        if potential_category not in unique_vals_col:
            new_category_name = potential_category
            break

    return new_category_name
