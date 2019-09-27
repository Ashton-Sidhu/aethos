import collections
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def label_encoder(list_of_cols=[], target=False, **datasets):
    """
    Label encodes the columns provided.
    
    Either the full data or training data plus testing data MUST be provided, not both.
    
    Parameters
    ----------
    list_of_cols : list, optional
        A list of specific columns to apply this technique to
        If `list_of_cols` is not provided, the strategy will be
        applied to all numeric columns., by default []

    target : bool, optional
            True if the column being encoded is the target variable, by default False

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
    
    label_encode = LabelEncoder()

    if data is not None:
        for col in list_of_cols:
            data[col] = label_encode.fit_transform(data[col])

        if target:
            target_mapping = dict(zip(data[list_of_cols], label_encode.inverse_transform(data[col])))
            target_mapping = OrderedDict(sorted(target_mapping.items()))

            return data, target_mapping

        return data
    else:
        for col in list_of_cols:
            x_train[col] = label_encode.fit_transform(x_train[col])
            x_test[col] = label_encode.transform(x_test[col])

        if target:
            target_mapping = dict(zip(x_train[list_of_cols], label_encode.inverse_transform(x_train[col])))
            target_mapping = OrderedDict(sorted(target_mapping.items()))

            return x_train, x_test, target_mapping

        return x_train, x_test

def check_missing_data(df) -> bool:
    """
    Utility function that checks if the data has any missing values.
    
    Parameters
    ----------
    df : Dataframe
        Pandas Dataframe of data
    
    Returns
    -------
    bool
        True if data has missing values, False otherwise.
    """   
    
    return df.isnull().values.any()

def get_keys_by_values(dict_of_elements: dict, item) -> list:
    """
    Utility function that returns the list of keys whos value matches a criteria defined
    by the param `item`.
    
    Parameters
    ----------
    dict_of_elements : dict
        Dictionary of key value mapping

    item : Any
        Value you want to return keys of.
    
    Returns
    -------
    list
        List of keys whos value matches the criteria defined by the param `item`
    """

    return [key for (key, value) in dict_of_elements.items() if value == item]


def drop_replace_columns(df, drop_cols, new_data, keep_col=False):
    """
    Utility function that drops a column that has been processed and replaces it with the new columns that have been derived from it.
    
    Parameters
    ----------
    df : Dataframe
        Dataframe of the data

    drop_cols : str or [str]
        Column or list of columns to be dropped

    new_data : Dataframe
        New data columns to be added to the dataframe
    
    Returns
    -------
    Dataframe
        Dataframe with the dropped column and the new data added
    """

    if keep_col:
        df = pd.concat([df, new_data], axis=1)
    else:
        df = df.drop(drop_cols, axis=1)
        df = pd.concat([df, new_data], axis=1)

    return df

def split_data(df, split_percentage: float):
    """
    Function that splits the data into a training and testing set. Split percentage is passed in through
    the split_percentage variable.
    
    Parameters
    ----------
    df : Dataframe
        Full dataset you want to split.

    split_percentage : float
        The % of data that you want in your test set, split_percentage is the percentage of data in the traning set.
    
    Returns
    -------
    Dataframe, Dataframe
        Train data and test data.
    """    

    x_train, x_test = train_test_split(df, test_size=split_percentage)

    return x_train, x_test

def _function_input_validation(data, x_train, x_test):
    """
    Helper function to help determine if input is valid.

    Unacceptable conditions
    _____________________

        1) No data is provided.
        2) Full dataset is provided and train or test data is provided.
        3) Train data is provided and test data is not.
        4) Test data is provided and test data is not.
    """

    if data is None and (x_train is None or x_test is None):
        return False

    if data is not None and (x_train is not None or x_test is not None):
        return False

    if x_train is not None and x_test is None:
        return False

    if x_test is not None and x_train is None:
        return False

    return True

def _numeric_input_conditions(list_of_cols: list, data, x_train) -> list:
    """
    Helper function to help set variable values of numeric cleaning method functions.

    If list of columns is provided, use it.

    If list of columns is not provided, use all the columns.
    """

    if list_of_cols:
        list_of_cols = list_of_cols
    else:
        if data is not None:
            list_of_cols = data.select_dtypes([np.number]).columns.tolist()
        else:
            list_of_cols = x_train.select_dtypes([np.number]).columns.tolist()

    return list_of_cols

def _get_columns(list_of_cols, data, x_train) -> list:
    """
    If the list of columns are optional and no columns are provided, the columns are set
    to all the columns in the data.
    
    Parameters
    ----------
    list_of_cols : list
        List of columns to apply method to.

    data : Dataframe or array like - 2d
        Full dataset

    x_train : Dataframe or array like - 2d
        Training Dataset
    
    Returns
    -------
    list
        List of columns to apply technique to
    """

    if list_of_cols:
        list_of_cols = list_of_cols
    else:
        if data is not None:
            list_of_cols = data.columns.tolist()
        else:
            list_of_cols = x_train.columns.tolist()

    return list_of_cols

def _input_columns(list_args: list, list_of_cols: list):
    """
    Takes columns inputted as arguments vs. columns passed as a list
    and returns a final column list.
    
    Parameters
    ----------
    list_args : list
        Input columns passed as args

    list_of_cols : list
        Hardcoded provided list of input columns.
    
    Returns
    -------
    list
        List of columns in a dataset
    """

    if list_of_cols or (not list_of_cols and not list_args):
        column_list = list_of_cols
    else:
        column_list = list(list_args)  

    return column_list

def _contructor_data_properties(step_obj):
    """
    Strips down a step object like Clean, Preprocess, Feature, etc and returns its data properties
    
    Parameters
    ----------
    step_obj : object
        Step object such as Clean, Preprocess, Feature

    Returns
    -------
    _data_properties: object
        Data object
    """

    if not step_obj:
        return None
    else:
        # Big hack until I implement a self __deepcopy__ implementation
        try:
            return step_obj._data_properties
        except:
            return step_obj

def _validate_model_name(model_obj, model_name: str) -> bool:
    """
    Validates the inputted model name. If the object already has an
    attribute with that model name, it is invalid
    
    Parameters
    ----------
    model_name : str
        Proposed name of the model
        
    model_obj : Model
        Model object
    
    Returns
    -------
    bool
        True if model name is valid, false otherwise
    """

    if hasattr(model_obj, model_name):
        return False

    return True

def _set_item(x_train, x_test, column: str, value: list, train_length: int, test_length: int):
    """
    Utility function for __setitem__ for determining which input is for which dataset
    and then sets the input to the new column for the correct dataset.
    
    Parameters
    ----------
    x_train : Dataframe
        Training Data

    x_test : Dataframe
        Testing Data

    column : str
        New column name

    value : list
        List of values for new column

    train_length : int
        Length of training data
        
    test_length : int
        Length of training data
    """

    ## If the training data and testing data have the same number of rows, apply the value to both
    ## train and test data set
    if len(value) == train_length and len(value) == test_length:
        x_train[column] = value
        x_test[column] = value
    elif len(value) == train_length:
        x_train[column] = value
    else:
        x_test[column] = value

    return x_train, x_test
