def process_text(
    corpus, lower=True, punctuation=True, stopwords=True, stemmer=True, numbers=True,
):
    """
    Function that takes text and does the following:

    - Casts it to lowercase
    - Removes punctuation
    - Removes stopwords
    - Stems the text
    - Removes any numerical values
    
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

    numbers : bool, optional
        True to remove any numbers, by default True
    
    Returns
    -------
    str
        Normalized text
    """

    import nltk
    import string
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import word_tokenize

    transformed_corpus = ""

    if lower:
        corpus = corpus.lower()

    for token in word_tokenize(corpus):

        if punctuation:
            if token in string.punctuation:
                continue

            token = token.translate(str.maketrans("", "", string.punctuation))

        if numbers:
            token = token.translate(str.maketrans("", "", "0123456789"))

        if stopwords:
            stop_words = nltk.corpus.stopwords.words("english")
            if token in stop_words:
                continue

        if stemmer:
            stem = SnowballStemmer("english")
            token = stem.stem(token)

        transformed_corpus += token + " "

    return transformed_corpus.strip()
