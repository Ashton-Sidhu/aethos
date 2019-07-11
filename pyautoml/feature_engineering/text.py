"""
This file contains the following methods:

FeatureBagOfWords
FeatureTFIDF
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from pyautoml.util import (DropAndReplaceColumns, GetListOfCols,
                           _FunctionInputValidation)

#TODO: Add customization to BoW and TF-IDF through the parameters of the constructor

def FeatureBagOfWords(list_of_cols=[], data=None, train_data=None, test_data=None, params={}):
    """Creates a matrix of how many times a word appears in a document.

    Either data or train_data or test_data MUST be provided, not both. 
    
    Keyword Arguments:
        list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
        data {DataFrame} or {[list]} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})
        params {dictionary} - Parameters you would pass into Bag of Words constructor as a dictionary

    Returns:
        [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
        of both are returned. 
    """

    if not _FunctionInputValidation(data, train_data, test_data):
        return "Function input is incorrectly provided."

    enc = CountVectorizer(**params)
    
    if isinstance(data, list):

        bag_of_words = enc.fit_transform(data)

        return bag_of_words
    
    if data is not None:
        for col in list_of_cols:
            enc_data = enc.fit_transform(df[col])
            enc_df = pd.DataFrame(enc_data, columns=enc_data.get_feature_names)
            data = DropAndReplaceColumns(data, col, enc_df)

        return data

    else:
        for col in list_of_cols:
            enc_train_data = enc.fit_transform(train_data[col])
            enc_train_df = pd.DataFrame(enc_train_data, columns=enc_train_data.get_feature_names)
            train_data = DropAndReplaceColumns(train_data, col, enc_train_df)

            enc_test_data = enc.transform(test_data[col])
            enc_test_df = pd.DataFrame(enc_test_data, columns=enc_test_data.get_features_names)
            test_data = DropAndReplaceColumns(test_data, col, enc_test_df)

        return train_data, test_data


def FeatureTFIDF(list_of_cols=[], data=None, train_data=None, test_data=None, params={}):
    """Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.

    Either data or train_data or test_data MUST be provided, not both. 
    
    Keyword Arguments:
        list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
        data {DataFrame} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})
        params {dictionary} - Parameters you would pass into Bag of Words constructor as a dictionary

    Returns:
        [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
        of both are returned. 
    """

    if not _FunctionInputValidation(data, train_data, test_data):
        return "Function input is incorrectly provided."
        
    enc = TfidfVectorizer(**params)

    if isinstance(data, list):

        tfidf = enc.fit_transform(data)

        return tfidf

    if data is not None:
        for col in list_of_cols:
            enc_data = enc.fit_transform(df[col])
            enc_df = pd.DataFrame(enc_data, columns=enc_data.get_feature_names)
            data = DropAndReplaceColumns(data, col, enc_df)

        return data

    else:
        for col in list_of_cols:
            enc_train_data = enc.fit_transform(train_data[col])
            enc_train_df = pd.DataFrame(enc_train_data, columns=enc_train_data.get_feature_names)
            train_data = DropAndReplaceColumns(train_data, col, enc_train_df)

            enc_test_data = enc.transform(test_data[col])
            enc_test_df = pd.DataFrame(enc_test_data, columns=enc_test_data.get_features_names)
            test_data = DropAndReplaceColumns(test_data, col, enc_test_df)

        return train_data, test_data
