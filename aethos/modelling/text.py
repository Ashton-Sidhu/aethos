import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize
from nltk.tokenize import word_tokenize

from aethos.preprocessing.text import process_text


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

    if prep:
        w2v = Word2Vec(
            sentences=[word_tokenize(process_text(text)) for text in x_train[col_name]],
            **algo_kwargs
        )
    else:
        w2v = Word2Vec(sentences=x_train[col_name], **algo_kwargs)

    return w2v


def gensim_doc2vec(x_train, x_test=None, prep=False, col_name=None, **algo_kwargs):
    """
    Uses Gensim Text Rank summarize to extract keywords.

    Note: this uses a variant of Text Rank.
    
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

    if prep:
        tagged_data = [
            TaggedDocument(words=word_tokenize(process_text(text)), tags=[str(i)])
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


def gensim_lda(x_train, x_test=None, prep=False, col_name=None, **algo_kwargs):
    """
    Runs Gensim LDA model and assigns topics to documents.
    
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
    Dataframe, *Dataframe, LDAModel, corpus, list
        Transformed dataframe with the new column.

    Returns 2 Dataframes if x_test is provided. 
    """

    texts = x_train[col_name].tolist()

    if prep:
        texts = [word_tokenize(process_text(text)) for text in texts]

    id2word = gensim.corpora.Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, **algo_kwargs)

    x_train["topics"] = _assign_topic_doc(lda_model, texts, corpus)

    if x_test is not None:
        texts = x_test[col_name].tolist()

        if prep:
            texts = [word_tokenize(process_text(text)) for text in texts]

        test_corpus = [id2word.doc2bow(text) for text in texts]

        x_test["topics"] = _assign_topic_doc(lda_model, texts, test_corpus)

    return x_train, x_test, lda_model, corpus, id2word


def _assign_topic_doc(lda_model, texts, corpus):
    """
    Helper function to assign the relevant topics to each document
    
    Parameters
    ----------
    lda_model : LDAModel
        LDA Model

    texts : [str]
        List of text documents

    corpus : list
        Corpus list
    
    Returns
    -------
    list
        List of topics assigned to each document
    """

    keywords = []

    for i, row in enumerate(lda_model[corpus]):

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                keywords.append(topic_keywords)

                break

    return keywords
