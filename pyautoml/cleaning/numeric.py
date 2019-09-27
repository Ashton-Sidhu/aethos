'''
This file contains the following methods:

replace_missing_mean_median_mode
replace_missing_constant
'''

import pandas as pd
from sklearn.impute import SimpleImputer

from pyautoml.cleaning.categorical import replace_missing_new_category
from pyautoml.util import (_function_input_validation, _get_columns,
                           _numeric_input_conditions, drop_replace_columns)

#TODO: Implement KNN, Interpolation, Extrapolation, Hot-Deck imputation for replacing missing data



def replace_missing_mean_median_mode(list_of_cols=[], strategy='', **datasets):
    """
    Replaces missing values in every numeric column with the mean, median or mode of that column specified by strategy.

    Mean: Average value of the column. Effected by outliers.
    Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.
    Mode: Most common number in a list of numbers.

    Either the full data or training data plus testing data MUST be provided, not both.
    
    Parameters
    ----------
    list_of_cols : list, optional
        A list of specific columns to apply this technique to
        If `list_of_cols` is not provided, the strategy will be
        applied to all numeric columns., by default []

    strategy : str
        Strategy for replacing missing values.
        Can be either "mean", "median" or "most_frequent"

    data: Dataframe or array like - 2d
        Full dataset, by default None.

    x_train: Dataframe or array like - 2d
        Training dataset, by default None.

    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    Returns 2 Dataframes if Train and Test data is provided.  
    """

    data = datasets.pop('data', None)
    x_train = datasets.pop('x_train', None)
    x_test = datasets.pop('x_test', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))

    if not _function_input_validation(data, x_train, x_test):
        raise ValueError("Function input is incorrectly provided.")
    
    if strategy != 'most_frequent':    
        list_of_cols = _numeric_input_conditions(list_of_cols, data, x_train)
    else:
        list_of_cols = _get_columns(list_of_cols, data, x_train)
    
    imp = SimpleImputer(strategy=strategy)
    
    if data is not None:                
        fit_data = imp.fit_transform(data[list_of_cols])
        fit_df = pd.DataFrame(fit_data, columns=list_of_cols)
        data = drop_replace_columns(data, list_of_cols, fit_df)

        return data
    else:
        fit_x_train = imp.fit_transform(x_train[list_of_cols])
        fit_train_df = pd.DataFrame(fit_x_train, columns=list_of_cols)            
        x_train = drop_replace_columns(x_train, list_of_cols, fit_train_df)
        
        fit_x_test = imp.transform(x_test[list_of_cols])
        fit_test_df = pd.DataFrame(fit_x_test, columns=list_of_cols)      
        x_test = drop_replace_columns(x_test, list_of_cols, fit_test_df)

        return x_train, x_test

def replace_missing_constant(col_to_constant=None, constant=0, **datasets):
    """
    Replaces missing values in every numeric column with a constant. If `col_to_constant` is not provided,
    all the missing values in the data will be replaced with `constant`
    
    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    col_to_constant : list, dict, optional
        Either a list of columns to replace missing values or a `column`: `value` dictionary mapping,
        by default None

    constant : int, float, optional
        Value to replace missing values with, by default 0

    data: Dataframe or array like - 2d
        Full dataset, by default None.

    x_train: Dataframe or array like - 2d
        Training dataset, by default None.
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    Returns 2 Dataframes if Train and Test data is provided.  
    
    Examples
    ------
    >>> replace_missing_constant({'a': 1, 'b': 2, 'c': 3})

    >>> replace_missing_constant(1, ['a', 'b', 'c'])
    """   

    data = datasets.pop('data', None)
    x_train = datasets.pop('x_train', None)
    x_test = datasets.pop('x_test', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))

    if not _function_input_validation(data, x_train, x_test):
        raise ValueError("Function input is incorrectly provided.")

    if isinstance(col_to_constant, dict):
        if data is not None:
            data = replace_missing_new_category(col_to_category=col_to_constant, data=data)

            return data
        
        else:
            x_train, x_test = replace_missing_new_category(col_to_cateogory=col_to_constant, x_train=x_train, x_test=x_test)

            return x_train, x_test

    elif isinstance(col_to_constant, list):
        if data is not None:
            data = replace_missing_new_category(constant=constant, col_to_category=col_to_constant, data=data)

            return data
        
        else:
            x_train, x_test = replace_missing_new_category(constant=constant, col_to_category=col_to_constant, x_train=x_train, x_test=x_test)

            return x_train, x_test

    else:
        if data is not None:
            data = replace_missing_new_category(constant=constant, data=data)

            return data
        
        else:
            x_train, x_test = replace_missing_new_category(constant=constant, x_train=x_train, x_test=x_test)

            return x_train, x_test
