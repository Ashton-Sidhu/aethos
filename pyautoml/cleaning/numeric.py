'''
This file contains the following methods:

replace_missing_mean_median_mode
replace_missing_constant
'''

import pandas as pd
from pyautoml.cleaning.categorical import replace_missing_new_category
from pyautoml.util import (_function_input_validation, _get_columns,
                           _numeric_input_conditions, drop_replace_columns)
from sklearn.impute import SimpleImputer

#TODO: Implement KNN, Interpolation, Extrapolation, Hot-Deck imputation for replacing missing data



def replace_missing_mean_median_mode(list_of_cols=[], strategy='', **datasets):
    """
    Replaces missing values in every numeric column with the mean, median or mode of that column specified by strategy.

    Mean: Average value of the column. Effected by outliers.
    Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.
    Mode: Most common number in a list of numbers.
    
    Parameters
    ----------
    list_of_cols : list, optional
        A list of specific columns to apply this technique to
        If `list_of_cols` is not provided, the strategy will be
        applied to all numeric columns., by default []
    strategy : str
        Strategy for replacing missing values.
        Can be either "mean", "median" or "most_frequent"

    Either the full data or training data plus testing data MUST be provided, not both.

    data: Dataframe or array like - 2d
        Full dataset, by default None.
    train_data: Dataframe or array like - 2d
        Training dataset, by default None.
    test_data: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    * Returns 2 Dataframes if Train and Test data is provided.  
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError(f"Invalid parameters passed: {str(datasets)}")

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")
    
    if strategy != 'most_frequent':    
        list_of_cols = _numeric_input_conditions(list_of_cols, data, train_data)
    else:
        list_of_cols = _get_columns(list_of_cols, data, train_data)
    
    imp = SimpleImputer(strategy=strategy)
    
    if data is not None:                
        fit_data = imp.fit_transform(data[list_of_cols])
        fit_df = pd.DataFrame(fit_data, columns=list_of_cols)
        data = drop_replace_columns(data, list_of_cols, fit_df)

        return data
    else:
        fit_train_data = imp.fit_transform(train_data[list_of_cols])
        fit_train_df = pd.DataFrame(fit_train_data, columns=list_of_cols)            
        train_data = drop_replace_columns(train_data, list_of_cols, fit_train_df)
        
        fit_test_data = imp.transform(test_data[list_of_cols])
        fit_test_df = pd.DataFrame(fit_test_data, columns=list_of_cols)      
        test_data = drop_replace_columns(test_data, list_of_cols, fit_test_df)

        return train_data, test_data

def replace_missing_constant(col_to_constant=None, constant=0, **datasets):
    """
    Replaces missing values in every numeric column with a constant. If `col_to_constant` is not provided,
    all the missing values in the data will be replaced with `constant`
    
    Parameters
    ----------
    col_to_constant : list, dict, optional
        Either a list of columns to replace missing values or a `column`: `value` dictionary mapping,
        by default None
    constant : int, float, optional
        Value to replace missing values with, by default 0
    
    Either the full data or training data plus testing data MUST be provided, not both.

    data: Dataframe or array like - 2d
        Full dataset, by default None.
    train_data: Dataframe or array like - 2d
        Training dataset, by default None.
    test_data: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    * Returns 2 Dataframes if Train and Test data is provided.  
    
    Examples
    ------
    >>> replace_missing_constant({'a': 1, 'b': 2, 'c': 3})

    >>> replace_missing_constant(1, ['a', 'b', 'c'])
    """   

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError(f"Invalid parameters passed: {str(datasets)}")

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    if isinstance(col_to_constant, dict):
        if data is not None:
            data = replace_missing_new_category(col_to_category=col_to_constant, data=data)

            return data
        
        else:
            train_data, test_data = replace_missing_new_category(col_to_cateogory=col_to_constant, train_data=train_data, test_data=test_data)

            return train_data, test_data

    elif isinstance(col_to_constant, list):
        if data is not None:
            data = replace_missing_new_category(constant=constant, col_to_category=col_to_constant, data=data)

            return data
        
        else:
            train_data, test_data = replace_missing_new_category(constant=constant, col_to_category=col_to_constant, train_data=train_data, test_data=test_data)

            return train_data, test_data

    else:
        if data is not None:
            data = replace_missing_new_category(constant=constant, data=data)

            return data
        
        else:
            train_data, test_data = replace_missing_new_category(constant=constant, train_data=train_data, test_data=test_data)

            return train_data, test_data
