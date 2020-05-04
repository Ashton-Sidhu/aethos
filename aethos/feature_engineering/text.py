import pandas as pd
import spacy

from textblob import TextBlob

from aethos.util import _get_columns


def textblob_features(
    x_train, x_test, feature, list_of_cols=[], new_col_name="_postagged",
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

    feature : str,
        Textblob feature

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

        if new_col_name.startswith("_"):
            new_col_name = col + new_col_name

        x_train[new_col_name] = pd.Series(
            [getattr(TextBlob(x), feature) for x in x_train[col]]
        )

        if x_test is not None:
            x_test[new_col_name] = pd.Series(
                [getattr(TextBlob(x), feature) for x in x_test[col]]
            )

    return x_train, x_test


def spacy_feature_postag(
    x_train, x_test=None, list_of_cols=[], new_col_name="_postagged", method="s"
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

    method : str {'s', 'd'}, optional
        Spacey PoS tagging method either simple or detailed
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    list_of_cols = _get_columns(list_of_cols, x_train)

    nlp = spacy.load("en_core_web_sm")

    if method == "s":
        func = lambda x: [(token, token.pos_) for token in x]
    else:
        func = lambda x: [(token, token.tag_) for token in x]

    for col in list_of_cols:

        if new_col_name.startswith("_"):
            new_col_name = col + new_col_name

        transformed_text = map(nlp, x_train[col])
        x_train[new_col_name] = pd.Series(map(func, transformed_text,))

        if x_test is not None:
            transformed_text = map(nlp, x_test[col])
            x_test[new_col_name] = pd.Series(map(func, transformed_text,))

    return x_train, x_test
