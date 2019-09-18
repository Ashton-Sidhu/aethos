"""
This file contains the following methods:

feature_one_hot_encode
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from pyautoml.util import (_function_input_validation, _get_columns,
                           drop_replace_columns)


def feature_one_hot_encode(list_of_cols: list, keep_col=True, **algo_kwargs):
    """
    Creates a matrix of converted categorical columns into binary columns of ones and zeros.
    
    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    list_of_cols : list
         A list of specific columns to apply this technique to.

    keep_col : bool
        A parameter to specify whether to drop the column being transformed, by default
        keep the column, True

    algo_kwargs : optional
        Parameters you would pass into Bag of Words constructor as a dictionary, by default {"handle_unknown": "ignore"}

    data : DataFrame
        Full dataset, by default None

    train_data : DataFrame
        Training dataset, by default None

    test_data : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if Train and Test data is provided. 
    """

    data = algo_kwargs.pop('data', None)
    train_data = algo_kwargs.pop('train_data', None)
    test_data = algo_kwargs.pop('test_data', None)

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    enc = OneHotEncoder(handle_unknown='ignore', **algo_kwargs)
    list_of_cols = _get_columns(list_of_cols, data, train_data)

    if data is not None:
        
        enc_data = enc.fit_transform(data[list_of_cols]).toarray()
        enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names(list_of_cols))
        data = drop_replace_columns(data, list_of_cols, enc_df, keep_col)

        return data

    else:        

        enc_train_data = enc.fit_transform(train_data[list_of_cols]).toarray()
        enc_train_df = pd.DataFrame(enc_train_data, columns=enc.get_feature_names(list_of_cols))
        train_data = drop_replace_columns(train_data, list_of_cols, enc_train_df, keep_col)

        enc_test_data = enc.transform(test_data[list_of_cols]).toarray()
        enc_test_df = pd.DataFrame(enc_test_data, columns=enc.get_feature_names(list_of_cols))
        test_data = drop_replace_columns(test_data, list_of_cols, enc_test_df, keep_col)

        return train_data, test_data
