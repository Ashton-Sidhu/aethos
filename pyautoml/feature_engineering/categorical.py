"""
This file contains the following methods:

feature_one_hot_encode
"""

import pandas as pd
from pyautoml.util import (_function_input_validation, drop_replace_columns,
                           get_list_of_cols)
from sklearn.preprocessing import OneHotEncoder


def feature_one_hot_encode(list_of_cols: list, params={"handle_unknown": "ignore"}, **datasets):
    """
    Creates a matrix of converted categorical columns into binary columns of ones and zeros.
    
    Parameters
    ----------
    list_of_cols : list
         A list of specific columns to apply this technique to.
    params : dict, optional
        Parameters you would pass into Bag of Words constructor as a dictionary, by default {"handle_unknown": "ignore"}

    Either the full data or training data plus testing data MUST be provided, not both.

    data : DataFrame
        Full dataset, by default None
    train_data : DataFrame
        Training dataset, by default None
    test_data : DataFrame
        Testing dataset, by default None
    
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

    enc = OneHotEncoder(**params)

    if data is not None:
        
        enc_data = enc.fit_transform(data[list_of_cols]).toarray()
        enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names().tolist())
        data = drop_replace_columns(data, list_of_cols, enc_df)

        return data

    else:        

        enc_train_data = enc.fit_transform(train_data[list_of_cols]).toarray()
        enc_train_df = pd.DataFrame(enc_train_data, columns=enc_data.get_feature_names().tolist())
        train_data = drop_replace_columns(train_data, list_of_cols, enc_train_df)

        enc_test_data = enc.transform(test_data[list_of_cols]).toarray()
        enc_test_df = pd.DataFrame(enc_test_data, columns=enc.get_features_names().tolist())
        test_data = drop_replace_columns(test_data, list_of_cols, enc_test_df)

        return train_data, test_data
