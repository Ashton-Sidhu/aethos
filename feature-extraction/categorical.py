import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from data.util import (DropAndReplaceColumns, GetListOfCols,
                       _FunctionInputValidation)


def FeatureOneHotEncode(list_of_cols, data=None, train_data=None, test_data=None, **onehot_kwargs):
    """Creates a matrix of converted categorical columns into binary columns.
    
    Either data or train_data or test_data MUST be provided, not both. 
    
    Keyword Arguments:
        list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
        data {DataFrame} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})
        **tfidf_kwargs {dictionary} - Parameters you would pass into Bag of Words constructor as a dictionary
    """

    if not _FunctionInputValidation(data, train_data, test_data):
        return "Function input is incorrectly provided."

    if not onehot_kwargs:
        onehot_kwargs = {"handle_unknown": "ignore"}

    enc = OneHotEncoder(**onehot_kwargs)

    if data is not None:
        
        enc_data = enc.fit_transform(data[list_of_cols]).toarray()
        enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names().tolist())
        data = DropAndReplaceColumns(data, list_of_cols, enc_df)

        return data

    else:        

        enc_train_data = enc.fit_transform(train_data[list_of_cols]).toarray()
        enc_train_df = pd.DataFrame(enc_train_data, columns=enc_data.get_feature_names().tolist())
        train_data = DropAndReplaceColumns(train_data, list_of_cols, enc_train_df)

        enc_test_data = enc.transform(test_data[list_of_cols]).toarray()
        enc_test_df = pd.DataFrame(enc_test_data, columns=enc.get_features_names().tolist())
        test_data = DropAndReplaceColumns(test_data, list_of_cols, enc_test_df)

        return train_data, test_data
