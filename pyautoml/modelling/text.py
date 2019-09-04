
from gensim.summarization import keywords, keywords.keywords
from gensim.summarization.summarizer import summarize


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

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        for col in list_of_cols:
            data.loc[:, col + new_col_name] = list(map(lambda x: summarize(x, **algo_kwargs), data[col]))

        return data
    else:
        for col in list_of_cols:
            train_data.loc[:, col + new_col_name] = list(map(lambda x: summarize(x, **algo_kwargs), train_data[col]))
            test_data.loc[:, col + new_col_name]  = list(map(lambda x: summarize(x, **algo_kwargs), test_data[col]))

        return train_data, test_data


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

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        for col in list_of_cols:
            data.loc[:, col + new_col_name] = list(map(lambda x: keywords.keywords(x, **algo_kwargs), data[col]))

        return data
    else:
        for col in list_of_cols:
            train_data.loc[:, col + new_col_name] = list(map(lambda x: keywords.keywords(x, **algo_kwargs), train_data[col]))
            test_data.loc[:, col + new_col_name]  = list(map(lambda x: keywords.keywords(x, **algo_kwargs), test_data[col]))

        return train_data, test_data


def gensim_textrank_generic_keywords(list_of_cols=[], new_col_name="_extracted_keywords", **datasets):
    """
    Uses Gensim Text Rank summarize to extractively summarize text.

    This is a generic version with no customizability.

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

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError(f'Invalid parameters passed: {str(datasets)}')

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

     if data is not None:
        for col in list_of_cols:
            data.loc[:, col + new_col_name] = list(map(lambda x: keywords(x), data[col]))

        return data
    else:
        for col in list_of_cols:
            train_data.loc[:, col + new_col_name] = list(map(lambda x: keywords(x), train_data[col]))
            test_data.loc[:, col + new_col_name]  = list(map(lambda x: keywords(x), test_data[col]))

        return train_data, test_data
