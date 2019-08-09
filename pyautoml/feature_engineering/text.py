"""
This file contains the following methods:

FeatureBagOfWords
FeatureTFIDF
"""

import pandas as pd
from pyautoml.util import (DropAndReplaceColumns, GetListOfCols,
                           _FunctionInputValidation)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob


def FeatureBagOfWords(list_of_cols=[], data=None, train_data=None, test_data=None, bow_params={}):
    """
    Creates a matrix of how many times a word appears in a document.

    Either data or train_data or test_data MUST be provided, not both.

    Keyword Arguments:
        list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
        data {DataFrame} or {[list]} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})
        bow_params {dictionary} - Parameters you would pass into Bag of Words constructor as a dictionary

    Returns:
        [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version
        of both are returned.
    """

    if not _FunctionInputValidation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    enc = CountVectorizer(**bow_params)

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
            enc_train_df = pd.DataFrame(
                enc_train_data, columns=enc_train_data.get_feature_names)
            train_data = DropAndReplaceColumns(train_data, col, enc_train_df)

            enc_test_data = enc.transform(test_data[col])
            enc_test_df = pd.DataFrame(
                enc_test_data, columns=enc_test_data.get_features_names)
            test_data = DropAndReplaceColumns(test_data, col, enc_test_df)

        return train_data, test_data


def FeatureTFIDF(list_of_cols=[], data=None, train_data=None, test_data=None, tfidf_params={}):
    """
    Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.

    Either data or train_data or test_data MUST be provided, not both.

    Keyword Arguments:
        list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
        data {DataFrame} -- Full dataset (default: {None})
        train_data {DataFrame} -- Training dataset (default: {None})
        test_data {DataFrame} -- Testing dataset (default: {None})
        tfidf_params {dictionary} - Parameters you would pass into Bag of Words constructor as a dictionary

    Returns:
        [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version
        of both are returned.
    """

    if not _FunctionInputValidation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    enc = TfidfVectorizer(tfidf_params)

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
            enc_train_df = pd.DataFrame(
                enc_train_data, columns=enc_train_data.get_feature_names)
            train_data = DropAndReplaceColumns(train_data, col, enc_train_df)

            enc_test_data = enc.transform(test_data[col])
            enc_test_df = pd.DataFrame(
                enc_test_data, columns=enc_test_data.get_features_names)
            test_data = DropAndReplaceColumns(test_data, col, enc_test_df)

        return train_data, test_data


def NLTKFeaturePoSTag(list_of_cols=[], data=None, train_data=None, test_data=None):    
    """
    Part of Speech tag the text data provided. Used to tag each word as a Noun, Adjective,
    Verbs, etc.

    This utilizes TextBlob which utlizes the NLTK tagger and is a wrapper for the tagging process.

    Args:
        list_of_cols (list, optional):  A list of specific columns to apply this technique to.. Defaults to [].
        data (DataFrame, optional): Full dataset. Defaults to None.
        train_data (Dataframe, optional): Training dataset. Defaults to None.
        test_data (Dataframe, optional): Test dataset. Defaults to None.
    
    Returns:
        Dataframe, *Dataframe: Transformed dataframe with new column of the text columns PoS tagged.

        * Returns 2 Dataframes if Train and Test data is provided.
    """

    if not _FunctionInputValidation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    if data is not None:
        for col in list_of_cols:
            data[col +
                 '_postagged'] = data[col].apply(lambda x: TextBlob(x).tags)

        return data

    else:
        for col in list_of_cols:
            train_data[col +
                       '_postagged'] = train_data[col].apply(lambda x: TextBlob(x).tags)
            test_data[col +
                      '_postagged'] = test_data[col].apply(lambda x: TextBlob(x).tags)

        return train_data, test_data
