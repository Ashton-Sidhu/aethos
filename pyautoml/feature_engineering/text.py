"""
This file contains the following methods:

feature_bag_of_words
feature_tfidf
nltk_feature_postag
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob

from pyautoml.util import (DropAndReplaceColumns, GetListOfCols,
                           _FunctionInputValidation)


def feature_bag_of_words(list_of_cols=[],  params={}, **datasets):
    """
    Creates a matrix of how many times a word appears in a document.
    
    Args:
        list_of_cols (list, optional): A list of specific columns to apply this technique to. Defaults to [].
        params (dict, optional): Parameters you would pass into Bag of Words constructor as a dictionary.
                                    Defaults to {}.
    
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
            enc_train_df = pd.DataFrame(
                enc_train_data, columns=enc_train_data.get_feature_names)
            train_data = DropAndReplaceColumns(train_data, col, enc_train_df)

            enc_test_data = enc.transform(test_data[col])
            enc_test_df = pd.DataFrame(
                enc_test_data, columns=enc_test_data.get_features_names)
            test_data = DropAndReplaceColumns(test_data, col, enc_test_df)

        return train_data, test_data


def feature_tfidf(list_of_cols=[], params={}, **datasets):
    """
    Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.

    Args:
        list_of_cols (list, optional): A list of specific columns to apply this technique to. Defaults to [].
        params (dict, optional): Parameters you would pass into TFIDF constructor as a dictionary. Defaults to {}.
    
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
            enc_train_df = pd.DataFrame(
                enc_train_data, columns=enc_train_data.get_feature_names)
            train_data = DropAndReplaceColumns(train_data, col, enc_train_df)

            enc_test_data = enc.transform(test_data[col])
            enc_test_df = pd.DataFrame(
                enc_test_data, columns=enc_test_data.get_features_names)
            test_data = DropAndReplaceColumns(test_data, col, enc_test_df)

        return train_data, test_data


def nltk_feature_postag(list_of_cols=[], **datasets):    
    """
    Part of Speech tag the text data provided. Used to tag each word as a Noun, Adjective,
    Verbs, etc.

    This utilizes TextBlob which utlizes the NLTK tagger and is a wrapper for the tagging process.

    Args:
        list_of_cols (list, optional):  A list of specific columns to apply this technique to. Defaults to [].

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
