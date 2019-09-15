"""
This file contains the following methods:

feature_bag_of_words
feature_tfidf
nltk_feature_postag
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob

from pyautoml.util import (_function_input_validation, _get_columns,
                           drop_replace_columns)


def feature_bag_of_words(list_of_cols=[], keep_col=False, **algo_kwargs):
    """
    Creates a matrix of how many times a word appears in a document.
    
    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    list_of_cols : list, optional
        A list of specific columns to apply this technique to., by default []

    algo_kwargs : dict, optional
        Parameters you would pass into Bag of Words constructor as a dictionary., by default {}

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

    enc = CountVectorizer(**algo_kwargs)
    list_of_cols = _get_columns(list_of_cols, data, train_data)

    if data is not None:
        for col in list_of_cols:
            enc_data = enc.fit_transform(data[col]).toarray()
            enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names())
            data = drop_replace_columns(data, col, enc_df, keep_col)

        return data

    else:
        for col in list_of_cols:
            enc_train_data = enc.fit_transform(train_data[col]).toarray()
            enc_train_df = pd.DataFrame(
                enc_train_data, columns=enc.get_feature_names())
            train_data = drop_replace_columns(train_data, col, enc_train_df, keep_col)

            enc_test_data = enc.transform(test_data[col]).toarray()
            enc_test_df = pd.DataFrame(
                enc_test_data, columns=enc.get_features_names())
            test_data = drop_replace_columns(test_data, col, enc_test_df, keep_col)

        return train_data, test_data


def feature_tfidf(list_of_cols=[], keep_col=True, **algo_kwargs):
    """
    Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.
    
    Either the full data or training data plus testing data MUST be provided, not both.
    
    Parameters
    ----------
    list_of_cols : list, optional
        A list of specific columns to apply this technique to, by default []
    algo_kwargs :  optional
        Parameters you would pass into TFIDF constructor, by default {}
    data : DataFrame
        Full dataset, by default None
    train_data : DataFrame
        Training dataset, by default None
    test_data : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column

    Returns 2 Dataframes if Train and Test data is provided. 
    """

    data = algo_kwargs.pop('data', None)
    train_data = algo_kwargs.pop('train_data', None)
    test_data = algo_kwargs.pop('test_data', None)

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    enc = TfidfVectorizer(**algo_kwargs)
    list_of_cols = _get_columns(list_of_cols, data, train_data)

    if data is not None:
        for col in list_of_cols:
            enc_data = enc.fit_transform(data[col]).toarray()
            enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names())
            data = drop_replace_columns(data, col, enc_df, keep_col)

        return data

    else:
        for col in list_of_cols:
            enc_train_data = enc.fit_transform(train_data[col]).toarray()
            enc_train_df = pd.DataFrame(
                enc_train_data, columns=enc.get_feature_names())
            train_data = drop_replace_columns(train_data, col, enc_train_df, keep_col)

            enc_test_data = enc.transform(test_data[col]).toarray()
            enc_test_df = pd.DataFrame(
                enc_test_data, columns=enc.get_feature_names())
            test_data = drop_replace_columns(test_data, col, enc_test_df, keep_col)

        return train_data, test_data


def nltk_feature_postag(list_of_cols=[], new_col_name='_postagged', **datasets):    
    """
    Part of Speech tag the text data provided. Used to tag each word as a Noun, Adjective,
    Verbs, etc.

    This utilizes TextBlob which utlizes the NLTK tagger and is a wrapper for the tagging process.
    
    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    list_of_cols : list, optional
        A list of specific columns to apply this technique to, by default []
    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_postagged`
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

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))    

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")

    list_of_cols = _get_columns(list_of_cols, data, train_data)

    if data is not None:
        for col in list_of_cols:
            data[col +
                 new_col_name] = list(map(lambda x: TextBlob(x).tags, data[col]))

        return data

    else:
        for col in list_of_cols:
            train_data[col +
                       new_col_name] = list(map(lambda x: TextBlob(x).tags, train_data[col]))
            test_data[col +
                      new_col_name] = list(map(lambda x: TextBlob(x).tags, test_data[col]))

        return train_data, test_data
