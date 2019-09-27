
from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize

from pyautoml.util import _function_input_validation


def gensim_textrank_summarizer(list_of_cols=[], new_col_name="_summarized", **algo_kwargs):
    """
    Uses Gensim Text Rank summarize to extractively summarize text.

    Note this uses a variant of Text Rank.
    
    Parameters
    ----------
    list_of_cols : list, optional
        Column name(s) of text data that you want to summarize

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `_summarized`
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if Train and Test data is provided. 
    """

    data = algo_kwargs.pop('data', None)
    x_train = algo_kwargs.pop('x_train', None)
    x_test = algo_kwargs.pop('x_test', None)

    if not _function_input_validation(data, x_train, x_test):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        for col in list_of_cols:
            data.loc[:, col + new_col_name] = list(map(lambda x: summarize(x, **algo_kwargs), data[col]))

        return data
    else:
        for col in list_of_cols:
            x_train.loc[:, col + new_col_name] = list(map(lambda x: summarize(x, **algo_kwargs), x_train[col]))
            x_test.loc[:, col + new_col_name]  = list(map(lambda x: summarize(x, **algo_kwargs), x_test[col]))

        return x_train, x_test


def gensim_textrank_keywords(list_of_cols=[], new_col_name="_extracted_keywords", **algo_kwargs):
    """
    Uses Gensim Text Rank summarize to extract keywords.

    Note this uses a variant of Text Rank.
    
    Parameters
    ----------
    list_of_cols : list, optional
        Column name(s) of text data that you want to summarize

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `_extracted_keywords`
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if Train and Test data is provided. 
    """

    data = algo_kwargs.pop('data', None)
    x_train = algo_kwargs.pop('x_train', None)
    x_test = algo_kwargs.pop('x_test', None)

    if not _function_input_validation(data, x_train, x_test):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        for col in list_of_cols:
            data.loc[:, col + new_col_name] = list(map(lambda x: keywords(x, **algo_kwargs), data[col]))

        return data
    else:
        for col in list_of_cols:
            x_train.loc[:, col + new_col_name] = list(map(lambda x: keywords(x, **algo_kwargs), x_train[col]))
            x_test.loc[:, col + new_col_name]  = list(map(lambda x: keywords(x, **algo_kwargs), x_test[col]))

        return x_train, x_test
