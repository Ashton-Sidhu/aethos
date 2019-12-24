import string
from functools import partial

import pandas as pd
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import PorterStemmer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize

NLTK_STEMMERS = {"porter": PorterStemmer(), "snowball": SnowballStemmer("english")}

NLTK_LEMMATIZERS = {"wordnet": WordNetLemmatizer()}


def split_sentences(x_train, x_test=None, list_of_cols=[], new_col_name="_sentences"):
    """
    Splits text by its sentences and then saves that list in a new column.

    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        Column name(s) of text data that you want to separate into sentences

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_sentences`

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column

    Returns 2 Dataframes if x_test is provided. 
    """

    for col in list_of_cols:
        x_train[col + new_col_name] = pd.Series(map(sent_tokenize, x_train[col]))
        if x_test is not None:
            x_test[col + new_col_name] = pd.Series(map(sent_tokenize, x_test[col]))

    return x_train, x_test


def nltk_stem(
    x_train, x_test=None, list_of_cols=[], stemmer="porter", new_col_name="_stemmed"
):
    """
    Stems all the text in a column.

    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        Column name(s) of text data that you want to stem

    stemmer : str, optional
        Type of NLTK stemmer to use, by default porter

        Current stemming implementations:
            - Porter

        For more information please refer to the NLTK stemming api https://www.nltk.org/api/nltk.stem.html

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_stemmed`

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    stem = NLTK_STEMMERS[stemmer]
    # Create partial for speed purposes
    func = partial(_apply_text_method, transformer=stem.stem)

    for col in list_of_cols:
        x_train[col + new_col_name] = pd.Series(map(func, x_train[col]))

        if x_test is not None:
            x_test[col + new_col_name] = pd.Series(map(func, x_test[col]))

    return x_train, x_test


def nltk_word_tokenizer(
    x_train, x_test=None, list_of_cols=[], regexp="", new_col_name="_tokenized"
):
    """
    Splits text into its words. 

    Default is by spaces but if a regex expression is provided, it will use that.

    Parameters
    ----------
    x_train : DataFrame
        Dataset
        
    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        Column name(s) of text data that you want to stem

    pattern : str, optional
        Regex pattern used to split words.

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_tokenized`

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    tokenizer = RegexpTokenizer(regexp)

    for col in list_of_cols:
        if not regexp:
            x_train[col + new_col_name] = pd.Series(map(word_tokenize, x_train[col]))

            if x_test is not None:
                x_test[col + new_col_name] = pd.Series(map(word_tokenize, x_test[col]))
        else:
            x_train[col + new_col_name] = pd.Series(
                map(tokenizer.tokenize, x_train[col])
            )

            if x_test is not None:
                x_test[col + new_col_name] = pd.Series(
                    map(tokenizer.tokenize, x_test[col])
                )

    return x_train, x_test


def nltk_remove_stopwords(
    x_train, x_test=None, list_of_cols=[], custom_stopwords=[], new_col_name="_rem_stop"
):
    """
    Removes stopwords following the nltk English stopwords list.

    A list of custom words can be provided as well, usually for domain specific words.

    Parameters
    ----------
    x_train : DataFrame
        Dataset
        
    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        Column name(s) of text data that you want to stem

    regexp : str, optional
        Regex pattern used to split words.

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_rem_words`

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    stop_words = stopwords.words("english")
    stop_words.extend(custom_stopwords)
    stop_list = set(stop_words)

    for col in list_of_cols:
        x_train[col + new_col_name] = list(
            map(
                lambda x: " ".join(
                    [word for word in word_tokenize(x.lower()) if word not in stop_list]
                ),
                x_train[col],
            )
        )

        if x_test is not None:
            x_test[col + new_col_name] = list(
                map(
                    lambda x: " ".join(
                        [
                            word
                            for word in word_tokenize(x.lower())
                            if word not in stop_list
                        ]
                    ),
                    x_test[col],
                )
            )

    return x_train, x_test


def remove_punctuation(
    x_train,
    x_test=None,
    list_of_cols=[],
    regexp="",
    exceptions=[],
    new_col_name="_rem_punct",
):
    """
    Removes punctuation from a string.

    Defaults to removing all punctuation, but if regex of punctuation is provided, it will remove them.

    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        Column name(s) of text data that you want to stem

    regexp : str, optional
        Regex pattern of punctuation to be removed.

    exceptions : list, optional
        List of punctuation to remove.

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `_rem_punct`

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    delete_punct = set(string.punctuation) - set(exceptions)
    tokenizer = RegexpTokenizer(regexp)

    for col in list_of_cols:
        if not regexp:
            x_train[col + new_col_name] = list(
                map(
                    lambda x: "".join(
                        [letter for letter in x if letter not in delete_punct]
                    ),
                    x_train[col],
                )
            )

            if x_test is not None:
                x_test[col + new_col_name] = list(
                    map(
                        lambda x: "".join(
                            [letter for letter in x if letter not in delete_punct]
                        ),
                        x_test[col],
                    )
                )
        else:
            x_train[col + new_col_name] = list(
                map(lambda x: " ".join(tokenizer.tokenize(x)), x_train[col])
            )

            if x_test is not None:
                x_test[col + new_col_name] = list(
                    map(lambda x: " ".join(tokenizer.tokenize(x)), x_test[col])
                )

    return x_train, x_test


def _apply_text_method(text_data, transformer=None):
    """
    Applies a text based method to the given data, for example
    a Lemmatizer, Stemmer, etc.

    Parameters
    ----------
    text_data : str
        Text data to transform
        
    transformer : Transformation object, optional
        trasnformer applied on the data, for example
        lemmatizer, stemmer, etc. , by default None

    Returns
    -------
    str
        Transformed data
    """

    transformed_text_data = ""

    for word in text_data.split():
        transformed_text_data += f"{transformer(word)} "

    return transformed_text_data.strip()


def process_text(corpus, lower=True, punctuation=True, stopwords=True, stemmer=True):
    """
    Function that takes text and does the following:

      - Casts it to lowercase
      - Removes punctuation
      - Removes stopwords
      - Stems the text
    
    Parameters
    ----------
    corpus : str
        Text
    
    lower : bool, optional
        True to cast all text to lowercase, by default True
    
    punctuation : bool, optional
        True to remove punctuation, by default True
    
    stopwords : bool, optional
        True to remove stop words, by default True
    
    stemmer : bool, optional
        True to stem the data, by default True
    
    Returns
    -------
    str
        Normalized text
    """

    import nltk

    transformed_corpus = ""

    if lower:
        corpus = corpus.lower()

    for token in word_tokenize(corpus):

        if punctuation:
            if token in string.punctuation:
                continue

        if stopwords:
            stop_words = nltk.corpus.stopwords.words("english")
            if token in stop_words:
                continue

        if stemmer:
            stem = SnowballStemmer("english")
            token = stem.stem(token)

        transformed_corpus += token + " "

    return transformed_corpus.strip()
