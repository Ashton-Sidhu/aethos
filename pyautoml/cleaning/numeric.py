'''
This file contains the following methods:

ReplaceMissingMeanMedianMode
ReplaceMissingConstant
'''

import pandas as pd
from pyautoml.cleaning.categorical import ReplaceMissingNewCategory
from pyautoml.util import (DropAndReplaceColumns, _FunctionInputValidation,
                           _NumericFunctionInputConditions)
from sklearn.impute import SimpleImputer

#TODO: Implement KNN, Interpolation, Extrapolation, Hot-Deck imputation for replacing missing data

def ReplaceMissingMeanMedianMode(strategy, list_of_cols=[], **datasets):
    """
    Replaces missing values in every numeric column with the mean, median or mode of that column specified by strategy.


    Mean: Average value of the column. Effected by outliers.
    Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.
    Mode: Most common number in a list of numbers.

    Args:
        strategy ([type]):  Strategy for replacing missing values.
                        Can be either "mean", "median" or "most_frequent"

        list_of_cols (list, optional): A list of specific columns to apply this technique to
                                        If `list_of_cols` is not provided, the strategy will be
                                        applied to all numeric columns.
        
        Either the full data or training data plus testing data MUST be provided, not both.

        data {DataFrame} -- Full dataset. Defaults to None.
        train_data {DataFrame} -- Training dataset. Defaults to None.
        test_data {DataFrame} -- Testing dataset. Defaults to None.

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
        raise ValueError("Function input is incorrectly provided.")
    
    list_of_cols = _NumericFunctionInputConditions(list_of_cols, data, train_data)
    imp = SimpleImputer(strategy=strategy)
    
    if data is not None:                
        fit_data = imp.fit_transform(data[list_of_cols])
        fit_df = pd.DataFrame(fit_data, columns=list_of_cols)
        data = DropAndReplaceColumns(data, list_of_cols, fit_df)

        return data
    else:
        fit_train_data = imp.fit_transform(train_data[list_of_cols])
        fit_train_df = pd.DataFrame(fit_data, columns=list_of_cols)            
        train_data = DropAndReplaceColumns(train_data, list_of_cols, fit_train_df)
        
        fit_test_data = imp.transform(test_data[list_of_cols])
        fit_test_df = pd.DataFrame(fit_test_data, columns=list_of_cols)      
        test_data = DropAndReplaceColumns(test_data, list_of_cols, fit_test_df)

        return train_data, test_data

def ReplaceMissingConstant(constant=0, col_to_constant=None, **datasets):

    """
    Replaces missing values in every numeric column with a constant. If `col_to_constant` is not provided,
    all the missing values in the data will be replaced with `constant`
    
    Args:
        constant (int or float, optional): [description]. Defaults to 0.
        col_to_constant (list or dict, optional): A list or dictionary of specific columns
         to apply this technique to. . Defaults to None.
    
        Either the full data or training data plus testing data MUST be provided, not both.

        data {DataFrame} -- Full dataset. Defaults to None.
        train_data {DataFrame} -- Training dataset. Defaults to None.
        test_data {DataFrame} -- Testing dataset. Defaults to None.
    
    Returns:
        Dataframe, *Dataframe: Transformed dataframe with rows with a missing values in a specific column are missing

        * Returns 2 Dataframes if Train and Test data is provided.

    Examples:

    >>>> ReplaceMissingConstant({'a': 1, 'b': 2, 'c': 3})
    >>>> ReplaceMissingConstant(1, ['a', 'b', 'c'])
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError(f"Invalid parameters passed: {str(datasets)}")

    if not _FunctionInputValidation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    if isinstance(col_to_constant, dict):
        if data is not None:

            data = ReplaceMissingNewCategory(col_to_category=col_to_constant, data=data)

            return data
        
        else:

            train_data, test_data = ReplaceMissingNewCategory(col_to_cateogory=col_to_constant, train_data=train_data, test_data=test_data)

            return train_data, test_data

    elif isinstance(col_to_constant, list):

        if data is not None:

            data = ReplaceMissingNewCategory(constant=constant, col_to_category=col_to_constant, data=data)

            return data
        
        else:

            train_data, test_data = ReplaceMissingNewCategory(constant=constant, col_to_category=col_to_constant, train_data=train_data, test_data=test_data)

            return train_data, test_data

    else:

        if data is not None:

            data = ReplaceMissingNewCategory(constant=constant, data=data)

            return data
        
        else:

            train_data, test_data = ReplaceMissingNewCategory(constant=constant, train_data=train_data, test_data=test_data)

            return train_data, test_data
