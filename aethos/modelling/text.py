from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize
from nltk.tokenize import word_tokenize


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
        x_train.loc[:, col + new_col_name] = [
            summarize(x, **algo_kwargs) for x in x_train[col]
        ]

        if x_test is not None:
            x_test.loc[:, col + new_col_name] = [
                summarize(x, **algo_kwargs) for x in x_test[col]
            ]

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
        x_train.loc[:, col + new_col_name] = [
            keywords(x, **algo_kwargs) for x in x_train[col]
        ]

        if x_test is not None:
            x_test.loc[:, col + new_col_name] = [
                keywords(x, **algo_kwargs) for x in x_test[col]
            ]

    return x_train, x_test


def gensim_word2vec(x_train, x_test=None, prep=False, col_name=None, **algo_kwargs):
    """
    Uses Gensim Text Rank summarize to extract keywords.

    Note this uses a variant of Text Rank.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    prep : bool, optional
        True to prep the text
        False if text is already prepped.
        By default, False

    col_name : str, optional
        Column name of text data that you want to summarize
        
    Returns
    -------
    Word2Vec
        Word2Vec model
    """

    # TODO: Add better default behaviour and transformation of passing in raw text
    if prep:
        w2v = Word2Vec(
            sentences=[word_tokenize(text.lower()) for text in x_train[col_name]],
            **algo_kwargs
        )
    else:
        w2v = Word2Vec(sentences=x_train[col_name], **algo_kwargs)

    return w2v


def gensim_doc2vec(x_train, x_test=None, prep=False, col_name=None, **algo_kwargs):
    """
    Uses Gensim Text Rank summarize to extract keywords.

    Note this uses a variant of Text Rank.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset

    x_test : DataFrame
        Testing dataset, by default None

    prep : bool, optional
        True to prep the text
        False if text is already prepped.
        By default, False

    col_name : str, optional
        Column name of text data that you want to summarize
    
    Returns
    -------
    Doc2Vec
        Doc2Vec Model
    """

    # TODO: Add better default behaviour and transformation of passing in raw text
    if prep:
        tagged_data = [
            TaggedDocument(words=word_tokenize(text.lower()), tags=[str(i)])
            for i, text in enumerate(x_train[col_name])
        ]
    else:
        tagged_data = [
            TaggedDocument(words=text, tags=[str(i)])
            for i, text in enumerate(x_train[col_name])
        ]

    d2v = Doc2Vec(tagged_data, **algo_kwargs)
    d2v.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    return d2v
