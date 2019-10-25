from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize


def gensim_textrank_summarizer(
    x_train, x_test=None, list_of_cols=[], new_col_name="_summarized", **algo_kwargs
):
    """
    Uses Gensim Text Rank summarize to extractively summarize text.

    Note this uses a variant of Text Rank.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        Column name(s) of text data that you want to summarize

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `_summarized`
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    for col in list_of_cols:
        x_train.loc[:, col + new_col_name] = list(
            map(lambda x: summarize(x, **algo_kwargs), x_train[col])
        )

        if x_test is not None:
            x_test.loc[:, col + new_col_name] = list(
                map(lambda x: summarize(x, **algo_kwargs), x_test[col])
            )

    return x_train, x_test


def gensim_textrank_keywords(
    x_train,
    x_test=None,
    list_of_cols=[],
    new_col_name="_extracted_keywords",
    **algo_kwargs
):
    """
    Uses Gensim Text Rank summarize to extract keywords.

    Note this uses a variant of Text Rank.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        Column name(s) of text data that you want to summarize

    new_col_name : str, optional
        New column name to be created when applying this technique, by default `_extracted_keywords`
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    for col in list_of_cols:
        x_train.loc[:, col + new_col_name] = list(
            map(lambda x: keywords(x, **algo_kwargs), x_train[col])
        )

        if x_test is not None:
            x_test.loc[:, col + new_col_name] = list(
                map(lambda x: keywords(x, **algo_kwargs), x_test[col])
            )

    return x_train, x_test
