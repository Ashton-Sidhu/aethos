"""
This file contains the following methods:

feature_bag_of_words
feature_hashing_vectorizer
feature_tfidf
nltk_feature_postag
"""

import pandas as pd
import spacy
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from textblob import TextBlob

from aethos.util import _get_columns, drop_replace_columns


def feature_bag_of_words(
    x_train, x_test=None, list_of_cols=[], keep_col=False, **algo_kwargs
):
    """
    Creates a matrix of how many times a word appears in a document.
    
    Parameters
    ----------
    x_train : DataFrame
        Training dataset, by default None
        
    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        A list of specific columns to apply this technique to., by default []

    keep_col : bool, optional
        True if you want to keep the columns passed, otherwise remove it.

    algo_kwargs : dict, optional
        Parameters you would pass into Bag of Words constructor as a dictionary., by default {}
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    enc = CountVectorizer(**algo_kwargs)
    list_of_cols = _get_columns(list_of_cols, x_train)

    for col in list_of_cols:
        enc_data = enc.fit_transform(x_train[col]).toarray()
        enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names())
        x_train = drop_replace_columns(x_train, col, enc_df, keep_col)

        if x_test is not None:
            enc_x_test = enc.transform(x_test[col]).toarray()
            enc_test_df = pd.DataFrame(enc_x_test, columns=enc.get_features_names())
            x_test = drop_replace_columns(x_test, col, enc_test_df, keep_col)

    return x_train, x_test


def feature_hash_vectorizer(
    x_train, x_test=None, list_of_cols=[], keep_col=True, **hashing_kwargs
):
    """
    Returns a hashed encoding of text data.
    
    Parameters
    ----------
    x_train : DataFrame
        Training dataset, by default None
        
    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        A list of specific columns to apply this technique to., by default []

    keep_col : bool, optional
        True if you want to keep the columns passed, otherwise remove it.

    hashing_kwargs : dict, optional
        Parameters you would pass into Hashing Vectorizer constructor, by default {}
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    enc = HashingVectorizer(**hashing_kwargs)
    list_of_cols = _get_columns(list_of_cols, x_train)

    for col in list_of_cols:
        enc_data = enc.fit_transform(x_train[col]).toarray()
        enc_df = pd.DataFrame(enc_data)
        x_train = drop_replace_columns(x_train, col, enc_df, keep_col)

        if x_test is not None:
            enc_x_test = enc.transform(x_test[col]).toarray()
            enc_test_df = pd.DataFrame(enc_x_test)
            x_test = drop_replace_columns(x_test, col, enc_test_df, keep_col)

    return x_train, x_test


def feature_tfidf(x_train, x_test=None, list_of_cols=[], keep_col=True, **algo_kwargs):
    """
    Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.
    
    Either the full data or training data plus testing data MUST be provided, not both.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset
        
    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        A list of specific columns to apply this technique to, by default []

    keep_col : bool, optional
        True if you want to keep the columns passed, otherwise remove it.

    algo_kwargs :  optional
        Parameters you would pass into TFIDF constructor, by default {}
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column

    Returns 2 Dataframes if x_test data is provided. 
    """

    enc = TfidfVectorizer(**algo_kwargs)
    list_of_cols = _get_columns(list_of_cols, x_train)

    for col in list_of_cols:
        enc_data = enc.fit_transform(x_train[col]).toarray()
        enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names())
        x_train = drop_replace_columns(x_train, col, enc_df, keep_col)

        if x_test is not None:
            enc_x_test = enc.transform(x_test[col]).toarray()
            enc_test_df = pd.DataFrame(enc_x_test, columns=enc.get_feature_names())
            x_test = drop_replace_columns(x_test, col, enc_test_df, keep_col)

    return x_train, x_test


def nltk_feature_postag(
    x_train, x_test=None, list_of_cols=[], new_col_name="_postagged"
):
    """
    Part of Speech tag the text data provided. Used to tag each word as a Noun, Adjective,
    Verbs, etc.

    This utilizes TextBlob which utlizes the NLTK tagger and is a wrapper for the tagging process.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        A list of specific columns to apply this technique to, by default []

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_postagged`
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    list_of_cols = _get_columns(list_of_cols, x_train)

    for col in list_of_cols:
        x_train[col + new_col_name] = pd.Series([TextBlob(x) for x in x_train[col]])

        if x_test is not None:
            x_test[col + new_col_name] = pd.Series(
                [TextBlob(x).tags for x in x_test[col]]
            )

    return x_train, x_test


def nltk_feature_noun_phrases(
    x_train, x_test=None, list_of_cols=[], new_col_name="_phrases"
):
    """
    Extracts noun phrases from the given text.

    This utilizes TextBlob which utlizes the NLTK NLP engine.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        A list of specific columns to apply this technique to, by default []

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_phrases`
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    list_of_cols = _get_columns(list_of_cols, x_train)

    for col in list_of_cols:
        x_train[col + new_col_name] = pd.Series(
            [TextBlob(x).noun_phrases for x in x_train[col]]
        )

        if x_test is not None:
            x_test[col + new_col_name] = pd.Series(
                [TextBlob(x).noun_phrases for x in x_train[col]]
            )

    return x_train, x_test


# TODO: Add simple and complex spacy pos tagging
def spacy_feature_postag(
    x_train, x_test=None, list_of_cols=[], new_col_name="_postagged"
):
    """
    Part of Speech tag the text data provided. Used to tag each word as a Noun, Adjective,
    Verbs, etc.

    This utilizes the spacy NLP engine.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        A list of specific columns to apply this technique to, by default []

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_postagged`
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    list_of_cols = _get_columns(list_of_cols, x_train)

    nlp = spacy.load("en_core_web_sm")

    for col in list_of_cols:
        transformed_text = map(nlp, x_train[col])
        x_train[col + new_col_name] = pd.Series(
            map(lambda x: [(token, token.pos_) for token in x], transformed_text,)
        )

        if x_test is not None:
            transformed_text = map(nlp, x_test[col])
            x_test[col + new_col_name] = pd.Series(
                map(lambda x: [(token, token.pos_) for token in x], transformed_text,)
            )

    return x_train, x_test


# TODO: Double check spacy noun phrase implementation
def spacy_feature_noun_phrases(
    x_train, x_test=None, list_of_cols=[], new_col_name="_phrases"
):
    """
    Extracts noun phrases from the given data.

    This utilizes the spacy NLP engine.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        A list of specific columns to apply this technique to, by default []

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_phrases`
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    list_of_cols = _get_columns(list_of_cols, x_train)

    nlp = spacy.load("en")

    for col in list_of_cols:
        transformed_text = list(map(nlp, x_train[col]))
        x_train[col + new_col_name] = pd.Series(
            map(lambda x: [str(phrase) for phrase in x.noun_chunks], transformed_text)
        )

        if x_test is not None:
            transformed_text = map(nlp, x_test[col])
            x_test[col + new_col_name] = pd.Series(
                map(
                    lambda x: [str(phrase) for phrase in x.noun_chunks],
                    transformed_text,
                )
            )

    return x_train, x_test
