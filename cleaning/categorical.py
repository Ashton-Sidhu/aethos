'''
This file contains the following methods:

ReplaceMissingNewCategory
ReplaceMissingRemoveRow
'''

import numpy as np
import pandas as pd

from data.util import DropAndReplaceColumns
from util import _FunctionInputValidation

#TODO: Implement KNN, and replacing with most common category 

def ReplaceMissingNewCategory(col_to_category=None, constant=None, data=None, train_data=None, test_data=None):
    """Replaces missing values in categorical column with its own category. The category name can be provided
    through the `new_category_name` parameter or if a category name is not provided, this function will assign 
    the name based off default categories.

    For numeric categorical columns default values are: -1, -999, -9999
    For string categorical columns default values are: "Other", "MissingDataCategory"
    
    Keyword Arguments:
        new_category_name {None, str, int, float} -- Category to replace missing values with (default: {None})
        custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
        override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                Example: if custom_cols is provided and override is true, the technique will only be applied
                                to the the columns in custom_cols (default: {False})
    """

    if not _FunctionInputValidation(data, train_data, test_data):
        return "Please provide a full data or training and testing data."
    
    str_missing_categories = ["Other", "Unknown", "MissingDataCategory"]
    num_missing_categories = [-1, -999, -9999]

    if isinstance(col_to_category, dict):
        
        for col in col_to_category.keys():

            data[col].fillna(col_to_category[col], inplace=True)

        return data

    elif isinstance(col_to_category, list) and constant is not None:

        for col in col_to_category:

            data[col].fillna(constant, inplace=True)

        return data

    elif isinstance(col_to_category, list) and constant is None:

        for col in col_to_category:

            if np.issubdtype(data[col].dtype, np.number):
                new_category_name = _DetermineDefaultCategory(data, col, num_missing_categories)
                data[col].fillna(new_category_name, inplace=True)
                #Convert numeric categorical column to integer
                data[col] = data[col].astype(int)
            else:
                new_category_name = _DetermineDefaultCategory(data, col, str_missing_categories)
                data[col].fillna(new_category_name, inplace=True)           

        return data

    elif col_to_category is None and constant is not None:

        data = data.fillna(constant)

        return data

    else:

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
   

def ReplaceMissingRemoveRow(cols_to_remove, data=None, train_data=None, test_data=None):
    """Remove rows where the value of a column for those rows is missing.
    
    Keyword Arguments:
        custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
        override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                Example: if custom_cols is provided and override is true, the technique will only be applied
                                to the the columns in custom_cols (default: {False})
    """
    if not _FunctionInputValidation(data, train_data, test_data):
        return "Please provide a full data or training and testing data."

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
