import string
from functools import partial

import pandas as pd
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import PorterStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from pyautoml.util import _function_input_validation

NLTK_STEMMERS = {
    'porter': PorterStemmer()
}

NLTK_LEMMATIZERS = {
    'wordnet': WordNetLemmatizer()
}


def split_sentences(list_of_cols=[], new_col_name='_sentences', **datasets):
    """
    Splits text by its sentences and then saves that list in a new column.

    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    list_of_cols : list, optional
        Column name(s) of text data that you want to separate into sentences

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_sentences`

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

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))
    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        for col in list_of_cols:
            data[col +
                 new_col_name] = pd.Series(map(sent_tokenize, data.loc[:, col]))

        return data
    else:
        for col in list_of_cols:
            train_data[col +
                       new_col_name] = pd.Series(map(sent_tokenize, train_data[col]))
            test_data[col +
                      new_col_name] = pd.Series(map(sent_tokenize, test_data[col]))

        return train_data, test_data


def nltk_stem(list_of_cols=[], stemmer='porter', new_col_name="_stemmed", **datasets):
    """
    Stems all the text in a column.

    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    list_of_cols : list, optional
        Column name(s) of text data that you want to stem

    stemmer : str, optional
        Type of NLTK stemmer to use, by default porter

        Current stemming implementations:
            - Porter

        For more information please refer to the NLTK stemming api https://www.nltk.org/api/nltk.stem.html

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_stemmed`

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
        raise ValueError('Function input is incorrectly provided.')

    stem = NLTK_STEMMERS[stemmer]
    # Create partial for speed purposes
    func = partial(_apply_text_method, transformer=stem.stem)

    if data is not None:
        for col in list_of_cols:
            data.loc[:, col + new_col_name] = list(map(func, data[col]))

        return data
    else:
        for col in list_of_cols:
            train_data.loc[:, col + new_col_name] = map(func, train_data[col])
            test_data.loc[:, col + new_col_name] = map(func, test_data[col])

        return train_data, test_data


def nltk_word_tokenizer(list_of_cols=[], regexp='', new_col_name="_tokenized", **datasets):
    """
    Splits text into its words. 

    Default is by spaces but if a regex expression is provided, it will use that.

    Parameters
    ----------
    list_of_cols : list, optional
        Column name(s) of text data that you want to stem

    pattern : str, optional
        Regex pattern used to split words.

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_tokenized`

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
        raise ValueError('Function input is incorrectly provided.')

    tokenizer = RegexpTokenizer(regexp)

    if data is not None:
        for col in list_of_cols:
            if not regexp:
                data[col + new_col_name] = list(map(word_tokenize, data[col]))
            else:
                data[col +
                     new_col_name] = list(map(tokenizer.tokenize, data[col]))

        return data
    else:
        for col in list_of_cols:
            if not regexp:
                train_data[col +
                           new_col_name] = list(map(word_tokenize, train_data[col]))
                test_data[col +
                          new_col_name] = list(map(word_tokenize, test_data[col]))
            else:
                train_data[col + new_col_name] = list(
                    map(tokenizer.tokenize, train_data[col]))
                test_data[col +
                          new_col_name] = list(map(tokenizer.tokenize, test_data[col]))

        return train_data, test_data


def nltk_remove_stopwords(list_of_cols=[], custom_stopwords=[], new_col_name='_rem_stop', **datasets):
    """
    Removes stopwords following the nltk English stopwords list.

    A list of custom words can be provided as well, usually for domain specific words.

    Parameters
    ----------
    list_of_cols : list, optional
        Column name(s) of text data that you want to stem

    regexp : str, optional
        Regex pattern used to split words.

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_rem_words`

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
        raise TypeError('Invalid parameters passed: {}'.format(str(datasets)))

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    stop_words = stopwords.words('english')
    stop_words.extend(custom_stopwords)
    stop_list = set(stop_words)

    if data is not None:
        for col in list_of_cols:
            data[col + new_col_name] = list(map(lambda x: " ".join(
                [word for word in word_tokenize(x.lower()) if word not in stop_list]), data[col]))

        return data
    else:
        for col in list_of_cols:
            train_data[col + new_col_name] = list(map(lambda x: " ".join(
                [word for word in word_tokenize(x.lower()) if word not in stop_list]), train_data[col]))
            test_data[col + new_col_name] = list(map(lambda x: " ".join(
                [word for word in word_tokenize(x.lower()) if word not in stop_list]), test_data[col]))

        return train_data, test_data


def remove_punctuation(list_of_cols=[], regexp='', exceptions=[], new_col_name='_rem_punct', **datasets):
    """
    Removes punctuation from a string.

    Defaults to removing all punctuation, but if regex of punctuation is provided, it will remove them.

    Parameters
    ----------
    list_of_cols : list, optional
        Column name(s) of text data that you want to stem

    regexp : str, optional
        Regex pattern of punctuation to be removed.

    exceptions : list, optional
        List of punctuation to remove.

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `_rem_punct`

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
        raise TypeError('Invalid parameters passed: {}'.format(str(datasets)))

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    delete_punct = set(string.punctuation) - set(exceptions)
    tokenizer = RegexpTokenizer(regexp)

    if data is not None:
        for col in list_of_cols:
            if not regexp:
                data[col + new_col_name] = list(map(lambda x: "".join(
                    [letter for letter in x if letter not in delete_punct]), data[col]))
            else:
                data[col + new_col_name] = list(
                    map(lambda x: " ".join(tokenizer.tokenize(x)), data[col]))

        return data
    else:
        for col in list_of_cols:
            if not regexp:
                train_data[col + new_col_name] = list(map(lambda x: "".join(
                    [letter for letter in x if letter not in delete_punct]), train_data[col]))
                test_data[col + new_col_name] = list(map(lambda x: "".join(
                    [letter for letter in x if letter not in delete_punct]), test_data[col]))
            else:
                train_data[col + new_col_name] = list(
                    map(lambda x: " ".join(tokenizer.tokenize(x)), train_data[col]))
                test_data[col + new_col_name] = list(
                    map(lambda x: " ".join(tokenizer.tokenize(x)), test_data[col]))

        return train_data, test_data


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

    transformed_text_data = ''

    for word in text_data.split():
        transformed_text_data += '{} '.format(transformer(word))

    return transformed_text_data.strip()
