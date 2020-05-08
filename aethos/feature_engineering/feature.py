import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import PolynomialFeatures

from aethos.feature_engineering import text
from aethos.feature_engineering import util
from aethos.util import (
    _input_columns,
    _get_columns,
    drop_replace_columns,
    _numeric_input_conditions,
)


class Feature(object):
    def onehot_encode(
        self, *list_args, list_of_cols=[], keep_col=True, **onehot_kwargs
    ):
        """
        Creates a matrix of converted categorical columns into binary columns of ones and zeros.

        For more info see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

        If a list of columns is provided use the list, otherwise use arguments.
    
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        keep_col : bool
            A parameter to specify whether to drop the column being transformed, by default
            keep the column, True

        categories : ‘auto’ or a list of array-like, default=’auto’
            Categories (unique values) per feature:

                ‘auto’ : Determine categories automatically from the training data.

                list : categories[i] holds the categories expected in the ith column. The passed categories should not mix strings and numeric values within a single feature, and should be sorted in case of numeric values.

            The used categories can be found in the categories_ attribute.

        drop : ‘first’ or a array-like of shape (n_features,), default=None
            Specifies a methodology to use to drop one of the categories per feature. This is useful in situations where perfectly collinear features cause problems, such as when feeding the resulting data into a neural network or an unregularized regression.

                None : retain all features (the default).

                ‘first’ : drop the first category in each feature. If only one category is present, the feature will be dropped entirely.

                array : drop[i] is the category in feature X[:, i] that should be dropped.

        sparsebool : default=True
            Will return sparse matrix if set True else will return an array.

        dtype : number type, default=np.float
            Desired dtype of output.

        handle_unknown: {‘error’, ‘ignore’}, default='ignore'
            Whether to raise an error or ignore if an unknown categorical feature is present during transform (default is to raise).
            When this parameter is set to ‘ignore’ and an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros.
            In the inverse transform, an unknown category will be denoted as None.

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.onehot_encode('col1', 'col2', 'col3')
        >>> data.onehot_encode('col1', 'col2', 'col3', drop='first')
        """

        # If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        enc = OneHotEncoder(handle_unknown="ignore", **onehot_kwargs)
        list_of_cols = _get_columns(list_of_cols, self.x_train)

        enc_data = enc.fit_transform(self.x_train[list_of_cols]).toarray()
        enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names(list_of_cols))
        self.x_train = drop_replace_columns(
            self.x_train, list_of_cols, enc_df, keep_col
        )

        if self.x_test is not None:
            enc_test = enc.transform(self.x_test[list_of_cols]).toarray()
            enc_test_df = pd.DataFrame(
                enc_test, columns=enc.get_feature_names(list_of_cols)
            )
            self.x_test = drop_replace_columns(
                self.x_test, list_of_cols, enc_test_df, keep_col
            )

        return self

    def tfidf(self, *list_args, list_of_cols=[], keep_col=True, **tfidf_kwargs):
        """
        Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.

        The higher the score the more important a word is to a document, the lower the score (relative to the other scores)
        the less important a word is to a document.

        For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        keep_col : bool, optional
            True if you want to keep the column(s) or False if you want to drop the column(s)

        encoding: str, default=’utf-8’
            If bytes or files are given to analyze, this encoding is used to decode.

        decode_error: {‘strict’, ‘ignore’, ‘replace’} (default=’strict’)
            Instruction on what to do if a byte sequence is given to analyze that contains characters not of the given encoding.
            By default, it is ‘strict’, meaning that a UnicodeDecodeError will be raised.
            Other values are ‘ignore’ and ‘replace’.

        strip_accents: {‘ascii’, ‘unicode’, None} (default=None)
            Remove accents and perform other character normalization during the preprocessing step.\
            ‘ascii’ is a fast method that only works on characters that have an direct ASCII mapping.
            ‘unicode’ is a slightly slower method that works on any characters.
            None (default) does nothing.

            Both ‘ascii’ and ‘unicode’ use NFKD normalization from unicodedata.normalize.

        lowercase: bool (default=True)
            Convert all characters to lowercase before tokenizing.

        preprocessor: callable or None (default=None)
            Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.
            Only applies if analyzer is not callable.

        tokenizer: callable or None (default=None)
            Override the string tokenization step while preserving the preprocessing and n-grams generation steps.
            Only applies if analyzer == 'word'.

        analyzer: str, {‘word’, ‘char’, ‘char_wb’} or callable
            Whether the feature should be made of word or character n-grams
            Option ‘char_wb’ creates character n-grams only from text inside word boundaries;
            n-grams at the edges of words are padded with space.

            If a callable is passed it is used to extract the sequence of features out of the raw, unprocessed input.

        stop_words: str {‘english’}, list, or None (default=None)
            If a string, it is passed to _check_stop_list and the appropriate stop list is returned.
            ‘english’ is currently the only supported string value.
            There are several known issues with ‘english’ and you should consider an alternative (see Using stop words).

            If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens. Only applies if analyzer == 'word'.

            If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms.

        token_pattern: str
            Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'.
            The default regexp selects tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).
        
        ngram_range: tuple (min_n, max_n), default=(1, 1)
            The lower and upper boundary of the range of n-values for different n-grams to be extracted.
            All values of n such that min_n <= n <= max_n will be used.
            For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
            Only applies if analyzer is not callable.
        
        max_df: float in range [0.0, 1.0] or int (default=1.0)
            When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
            If float, the parameter represents a proportion of documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.

        min_df: float in range [0.0, 1.0] or int (default=1)
            When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
            This value is also called cut-off in the literature.
            If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.

        max_features: int or None (default=None)
            If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

            This parameter is ignored if vocabulary is not None.

        vocabulary: Mapping or iterable, optional (default=None)
            Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms.
            If not given, a vocabulary is determined from the input documents.

        binary: bool (default=False)
            If True, all non-zero term counts are set to 1.
            This does not mean outputs will have only 0/1 values, only that the tf term in tf-idf is binary. (Set idf and normalization to False to get 0/1 outputs).

        dtype: type, optional (default=float64)
            Type of the matrix returned by fit_transform() or transform().

        norm: ‘l1’, ‘l2’ or None, optional (default=’l2’)
            Each output row will have unit norm, either: * ‘l2’: Sum of squares of vector elements is 1.
            The cosine similarity between two vectors is their dot product when l2 norm has been applied. * ‘l1’: Sum of absolute values of vector elements is 1.

        use_idf: bool (default=True)
            Enable inverse-document-frequency reweighting.

        smooth_idf: bool (default=True)
            Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once.
            Prevents zero divisions.

        sublinear_tf: bool (default=False)
            Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.tfidf('col1', 'col2', 'col3')
        >>> data.tfidf('col1', 'col2', 'col3', lowercase=False, smoothidf=False)
        """

        # If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        enc = TfidfVectorizer(**tfidf_kwargs)
        list_of_cols = _get_columns(list_of_cols, self.x_train)

        for col in list_of_cols:
            enc_data = enc.fit_transform(self.x_train[col]).toarray()
            enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names())
            self.x_train = drop_replace_columns(self.x_train, col, enc_df, keep_col)

            if self.x_test is not None:
                enc_test = enc.transform(self.x_test[col]).toarray()
                enc_test_df = pd.DataFrame(enc_test, columns=enc.get_feature_names())
                self.x_test = drop_replace_columns(
                    self.x_test, col, enc_test_df, keep_col
                )

        return self

    def bag_of_words(self, *list_args, list_of_cols=[], keep_col=True, **bow_kwargs):
        """
        Creates a matrix of how many times a word appears in a document.

        The premise is that the more times a word appears the more the word represents that document.

        For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        keep_col : bool, optional
            True if you want to keep the column(s) or False if you want to drop the column(s)

        encoding: str, default=’utf-8’
            If bytes or files are given to analyze, this encoding is used to decode.

        decode_error: {‘strict’, ‘ignore’, ‘replace’} (default=’strict’)
            Instruction on what to do if a byte sequence is given to analyze that contains characters not of the given encoding.
            By default, it is ‘strict’, meaning that a UnicodeDecodeError will be raised.
            Other values are ‘ignore’ and ‘replace’.

        strip_accents: {‘ascii’, ‘unicode’, None} (default=None)
            Remove accents and perform other character normalization during the preprocessing step.\
            ‘ascii’ is a fast method that only works on characters that have an direct ASCII mapping.
            ‘unicode’ is a slightly slower method that works on any characters.
            None (default) does nothing.

            Both ‘ascii’ and ‘unicode’ use NFKD normalization from unicodedata.normalize.

        lowercase: bool (default=True)
            Convert all characters to lowercase before tokenizing.

        preprocessor: callable or None (default=None)
            Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.
            Only applies if analyzer is not callable.

        tokenizer: callable or None (default=None)
            Override the string tokenization step while preserving the preprocessing and n-grams generation steps.
            Only applies if analyzer == 'word'.

        analyzer: str, {‘word’, ‘char’, ‘char_wb’} or callable
            Whether the feature should be made of word or character n-grams
            Option ‘char_wb’ creates character n-grams only from text inside word boundaries;
            n-grams at the edges of words are padded with space.

            If a callable is passed it is used to extract the sequence of features out of the raw, unprocessed input.

        stop_words: str {‘english’}, list, or None (default=None)
            If a string, it is passed to _check_stop_list and the appropriate stop list is returned.
            ‘english’ is currently the only supported string value.
            There are several known issues with ‘english’ and you should consider an alternative (see Using stop words).

            If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens. Only applies if analyzer == 'word'.

            If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms.

        token_pattern: str
            Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'.
            The default regexp selects tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).
        
        ngram_range: tuple (min_n, max_n), default=(1, 1)
            The lower and upper boundary of the range of n-values for different n-grams to be extracted.
            All values of n such that min_n <= n <= max_n will be used.
            For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
            Only applies if analyzer is not callable.
        
        max_df: float in range [0.0, 1.0] or int (default=1.0)
            When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
            If float, the parameter represents a proportion of documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.

        min_df: float in range [0.0, 1.0] or int (default=1)
            When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
            This value is also called cut-off in the literature.
            If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.

        max_features: int or None (default=None)
            If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

            This parameter is ignored if vocabulary is not None.

        vocabulary: Mapping or iterable, optional (default=None)
            Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms.
            If not given, a vocabulary is determined from the input documents.

        binary: bool (default=False)
            If True, all non-zero term counts are set to 1.
            This does not mean outputs will have only 0/1 values, only that the tf term in tf-idf is binary. (Set idf and normalization to False to get 0/1 outputs).
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.bag_of_words('col1', 'col2', 'col3')
        >>> data.bag_of_words('col1', 'col2', 'col3', binary=True)
        """

        # If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        enc = CountVectorizer(**bow_kwargs)
        list_of_cols = _get_columns(list_of_cols, self.x_train)

        for col in list_of_cols:
            enc_data = enc.fit_transform(self.x_train[col]).toarray()
            enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names())
            self.x_train = drop_replace_columns(self.x_train, col, enc_df, keep_col)

            if self.x_test is not None:
                enc_test = enc.transform(self.x_test[col]).toarray()
                enc_test_df = pd.DataFrame(enc_test, columns=enc.get_feature_names())
                self.x_test = drop_replace_columns(
                    self.x_test, col, enc_test_df, keep_col
                )

        return self

    def text_hash(self, *list_args, list_of_cols=[], keep_col=True, **hash_kwargs):
        """
        Creates a matrix of how many times a word appears in a document. It can possibly normalized as token frequencies if norm=’l1’ or projected on the euclidean unit sphere if norm=’l2’.

        The premise is that the more times a word appears the more the word represents that document.

        This text vectorizer implementation uses the hashing trick to find the token string name to feature integer index mapping.

        This strategy has several advantages:

            It is very low memory scalable to large datasets as there is no need to store a vocabulary dictionary in memory
            It is fast to pickle and un-pickle as it holds no state besides the constructor parameters
            It can be used in a streaming (partial fit) or parallel pipeline as there is no state computed during fit.

        For more info please see: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html

        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        keep_col : bool, optional
            True if you want to keep the column(s) or False if you want to drop the column(s)

        n_features : integer, default=(2 ** 20)
            The number of features (columns) in the output matrices.
            Small numbers of features are likely to cause hash collisions, but large numbers will cause larger coefficient dimensions in linear learners.

        hash_kwargs : dict, optional
            Parameters you would pass into Bag of Words constructor, by default {}
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.text_hash('col1', 'col2', 'col3')
        >>> data.text_hash('col1', 'col2', 'col3', n_features=50)
        """

        # If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        enc = HashingVectorizer(**hash_kwargs)
        list_of_cols = _get_columns(list_of_cols, self.x_train)

        for col in list_of_cols:
            enc_data = enc.fit_transform(self.x_train[col]).toarray()
            enc_df = pd.DataFrame(enc_data)
            self.x_train = drop_replace_columns(self.x_train, col, enc_df, keep_col)

            if self.x_test is not None:
                enc_test = enc.transform(self.x_test[col]).toarray()
                enc_test_df = pd.DataFrame(enc_test)
                self.x_test = drop_replace_columns(
                    self.x_test, col, enc_test_df, keep_col
                )

        return self

    def postag_nltk(self, *list_args, list_of_cols=[], new_col_name="_postagged"):
        """
        Tag documents with their respective "Part of Speech" tag with the Textblob package which utilizes the NLTK NLP engine and Penn Treebank tag set.
        These tags classify a word as a noun, verb, adjective, etc. A full list and their meaning can be found here:
        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_postagged`

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.postag_nltk('col1', 'col2', 'col3')
        """

        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = text.textblob_features(
            x_train=self.x_train,
            x_test=self.x_test,
            feature="tags",
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
        )

        return self

    def postag_spacy(self, *list_args, list_of_cols=[], new_col_name="_postagged"):
        """
        Tag documents with their respective "Part of Speech" tag with the Spacy NLP engine and the Universal Dependencies scheme.
        These tags classify a word as a noun, verb, adjective, etc. A full list and their meaning can be found here:
        https://spacy.io/api/annotation#pos-tagging 

        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_postagged`

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.postag_spacy('col1', 'col2', 'col3')
        """

        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = text.spacy_feature_postag(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
            method="s",
        )

        return self

    def postag_spacy_detailed(
        self, *list_args, list_of_cols=[], new_col_name="_postagged"
    ):
        """
        Tag documents with their respective "Part of Speech" tag with the Spacy NLP engine and the PennState PoS tags.
        These tags classify a word as a noun, verb, adjective, etc. A full list and their meaning can be found here:
        https://spacy.io/api/annotation#pos-tagging 

        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_postagged`

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.postag_spacy_detailed('col1', 'col2', 'col3')
        """

        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = text.spacy_feature_postag(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
            method="d",
        )

        return self

    def nounphrases_nltk(self, *list_args, list_of_cols=[], new_col_name="_phrases"):
        """
        Extract noun phrases from text using the Textblob packages which uses the NLTK NLP engine.

        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_phrases`

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.nounphrases_nltk('col1', 'col2', 'col3')
        """

        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = text.textblob_features(
            x_train=self.x_train,
            x_test=self.x_test,
            feature="noun_phrases",
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
        )

        return self

    def nounphrases_spacy(self, *list_args, list_of_cols=[], new_col_name="_phrases"):
        """
        Extract noun phrases from text using the Textblob packages which uses the NLTK NLP engine.

        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_phrases`

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.nounphrases_spacy('col1', 'col2', 'col3')
        """

        import spacy

        list_of_cols = _input_columns(list_args, list_of_cols)
        list_of_cols = _get_columns(list_of_cols, self.x_train)

        nlp = spacy.load("en")

        for col in list_of_cols:

            if new_col_name.startswith("_"):
                new_col_name = col + new_col_name

            transformed_text = list(map(nlp, self.x_train[col]))
            self.x_train[new_col_name] = pd.Series(
                map(
                    lambda x: [str(phrase) for phrase in x.noun_chunks],
                    transformed_text,
                )
            )

            if self.x_test is not None:
                transformed_text = map(nlp, self.x_test[col])
                self.x_test[new_col_name] = pd.Series(
                    map(
                        lambda x: [str(phrase) for phrase in x.noun_chunks],
                        transformed_text,
                    )
                )

        return self

    def polynomial_features(self, *list_args, list_of_cols=[], **poly_kwargs):
        """
        Generate polynomial and interaction features.

        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree.
        
        For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        degree : int
            Degree of the polynomial features, by default 2

        interaction_only : boolean, 
            If true, only interaction features are produced: features that are products of at most degree distinct input features (so not x[1] ** 2, x[0] * x[2] ** 3, etc.).
            by default = False
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.polynomial_features('col1', 'col2', 'col3')
        """

        # If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        poly = PolynomialFeatures(**poly_kwargs)
        list_of_cols = _numeric_input_conditions(list_of_cols, self.x_train)

        scaled_data = poly.fit_transform(self.x_train[list_of_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=poly.get_feature_names())
        self.x_train = drop_replace_columns(self.x_train, list_of_cols, scaled_df)

        if self.x_test is not None:
            scaled_test = poly.transform(self.x_test)
            scaled_test_df = pd.DataFrame(scaled_test, columns=poly.get_feature_names())
            self.x_test = drop_replace_columns(
                self.x_test, list_of_cols, scaled_test_df
            )

        return self

    def apply(self, func, output_col: str):
        """
        Calls pandas apply function. Will apply the function to your dataset, or
        both your training and testing dataset.
        
        Parameters
        ----------
        func : Function pointer
            Function describing the transformation for the new column

        output_col : str
            New column name
        
        Returns
        -------
        Feature:
            Returns a deep copy of the Feature object.

        Examples
        --------
        >>>     col1  col2  col3 
            0     1     0     1       
            1     0     2     0       
            2     1     0     1
        >>> data.apply(lambda x: x['col1'] > 0, 'col4')
        >>>     col1  col2  col3  col4 
            0     1     0     1     1       
            1     0     2     0     0  
            2     1     0     1     1
        """

        import swifter

        self.x_train.loc[:, output_col] = self.x_train.swifter.progress_bar().apply(
            func, axis=1
        )

        if self.x_test is not None:
            self.x_test.loc[:, output_col] = self.x_test.swifter.progress_bar().apply(
                func, axis=1
            )

        return self

    def ordinal_encode_labels(self, col: str, ordered_cat=[]):
        """
        Encode categorical values with value between 0 and n_classes-1.

        Running this function will automatically set the corresponding mapping for the target variable mapping number to the original value.

        Note: that this will not work if your test data will have labels that your train data does not.
        Note: 

        Parameters
        ----------
        col : str, optional
            Columnm in the data to ordinally encode.

        ordered_cat : list, optional
            A list of ordered categories for the Ordinal encoder. Should be sorted.
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.encode_labels('col1')
        >>> data.encode_labels('col1', ordered_cat=["Low", "Medium", "High"])
        """

        categories = ordered_cat if ordered_cat else "auto"

        enc = OrdinalEncoder(categories=categories)

        self.x_train[col] = enc.fit_transform(self.x_train[col].values.reshape(-1, 1))

        if self.x_test is not None:
            self.x_test[col] = enc.transform(self.x_test[col].values.reshape(-1, 1))

        return self

    def pca(self, n_components=10, **pca_kwargs):
        """
        Reduces the dimensionality of the data using Principal Component Analysis. 
        
        Use PCA when the data is dense.

        This can be used to reduce complexity as well as speed up computation.

        For more info please see: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html 

        This function exists in `feature-extraction/util.py`
        
        Parameters
        ----------
        n_components : int, float, None or string, by default 10
            Number of components to keep.
            
            If n_components is not set all components are kept.
            If n_components == 'mle' and svd_solver == 'full', Minka’s MLE is used to guess the dimension.
            Use of n_components == 'mle' will interpret svd_solver == 'auto' as svd_solver == 'full'
            If 0 < n_components < 1 and svd_solver == 'full', select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components
            If svd_solver == 'arpack', the number of components must be strictly less than the minimum of n_features and n_samples

        whiten : bool, optional (default False)
            When True (False by default) the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
            Whitening will remove some information from the transformed signal (the relative variance scales of the components) but can sometime improve the predictive accuracy of the downstream estimators by making their data respect some hard-wired assumptions.

        svd_solver : string {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
            auto :
                the solver is selected by a default policy based on X.shape and n_components: if the input data is larger than 500x500 and the number of components to extract is lower than 80% of the smallest dimension of the data, then the more efficient ‘randomized’ method is enabled. Otherwise the exact full SVD is computed and optionally truncated afterwards.
            full :
                run exact full SVD calling the standard LAPACK solver via scipy.linalg.svd and select the components by postprocessing
            arpack :
                run SVD truncated to n_components calling ARPACK solver via scipy.sparse.linalg.svds. It requires strictly 0 < n_components < min(X.shape)
            randomized :
                run randomized SVD by the method of Halko et al.

        tol : float >= 0, optional (default .0)
            Tolerance for singular values computed by svd_solver == ‘arpack’.

        iterated_power : int >= 0, or ‘auto’, (default ‘auto’)
            Number of iterations for the power method computed by svd_solver == ‘randomized’.

        random_state : int, RandomState instance or None, optional (default None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
            Used when svd_solver == ‘arpack’ or ‘randomized’.
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.pca(n_components=2)
        """

        self._run_sklearn_dim_reduction("pca", n_components=n_components, **pca_kwargs)

        return self

    def truncated_svd(self, n_components=50, **svd_kwargs):
        """
        Reduces the dimensionality of the data using Truncated SVD.

        In particular, truncated SVD works on term count/tf-idf matrices.
        In that context, it is known as latent semantic analysis (LSA).
        
        Use Truncated SVD when the data is sparse.

        This can be used to reduce complexity as well as speed up computation.

        For more info please see: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        This function exists in `feature-extraction/util.py`
        
        Parameters
        ---------- 
        n_components: int, default = 2
            Desired dimensionality of output data. Must be strictly less than the number of features.
            The default value is useful for visualisation. For LSA, a value of 100 is recommended.

        algorithm: string, default = “randomized”
            SVD solver to use. Either “arpack” for the ARPACK wrapper in SciPy (scipy.sparse.linalg.svds), or “randomized” for the randomized algorithm due to Halko (2009).

        n_iter: int, optional (default 5)
            Number of iterations for randomized SVD solver. Not used by ARPACK.
            The default is larger than the default in ~sklearn.utils.extmath.randomized_svd to handle sparse matrices that may have large slowly decaying spectrum.

        tol: float, optional
            Tolerance for ARPACK. 0 means machine precision. Ignored by randomized SVD solver.
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.truncated_svd(n_components=2)
        """

        self._run_sklearn_dim_reduction("tsvd", n_components=n_components, **svd_kwargs)

        return self

    def drop_correlated_features(self, threshold=0.95):
        """
        Drop features that have a correlation coefficient greater than the specified threshold with other features.
        
        Parameters
        ----------
        threshold : float, optional
            Correlation coefficient threshold, by default 0.95

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.drop_correlated_features(threshold=0.9)
        """

        corr = self.x_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]

        self.x_train.drop(drop_cols, axis=1, inplace=True)

        if self.x_test is not None:
            self.x_test.drop(drop_cols, axis=1, inplace=True)

        return self

    def chi2_feature_selection(self, k: int, verbose=False):
        """
        Uses Chi2 to choose the best K features.

        The Chi2 null hypothesis is that 2 variables are independent.

        Chi-square test feature selection “weeds out” the features that are the most likely to be independent of class and therefore irrelevant for classification.
        
        Parameters
        ----------
        k : int or "all"
            Number of features to keep.

        verbose : bool
            True to print p-values for each feature, by default False
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.chi2_feature_selection(k=10)
        """

        y_train = self.y_train
        y_test = self.y_test
        self.x_train = self.x_train.drop(self.target, axis=1)
        self.x_test = (
            None if self.x_test is None else self.x_test.drop(self.target, axis=1)
        )

        chi2_best = SelectKBest(chi2, k=k).fit(self.x_train, y_train)

        column_indices = chi2_best.get_support()

        if verbose:
            for col, p in zip(
                self.x_train.columns[column_indices], chi2_best.pvalues_[column_indices]
            ):
                print(f"{col} p-value: {p}")

        self.x_train = self.x_train[self.x_train.columns[column_indices]]

        if self.x_test is not None:
            self.x_test = self.x_test[self.x_test.columns[column_indices]]

        # Re add the target field back to the dataset.
        self.x_train[self.target] = y_train
        if self.x_test is not None:
            self.x_test[self.target] = y_test

        return self

    def _run_sklearn_dim_reduction(self, algo: str, n_components, **kwargs):
        """
        Generalized helper function to run dimensionality reduction algorithms from sklearn.
        
        Parameters
        ----------
        algo : str {'pca', 'tsne', 'lle', 'tsvd'}
            Dim Reduction algorithm to run.
        """

        if self.target:
            train_target_data = self.x_train[self.target]
            test_target_data = (
                self.x_test[self.target] if self.x_test is not None else None
            )
            self.x_train = self.x_train.drop(self.target, axis=1)
            self.x_test = (
                self.x_test.drop(self.target, axis=1)
                if self.x_test is not None
                else None
            )
        else:
            self.x_train = self.x_train
            self.x_test = self.x_test

        self.x_train, self.x_test = util.sklearn_dim_reduction(
            x_train=self.x_train,
            x_test=self.x_test,
            algo=algo,
            n_components=n_components,
            **kwargs,
        )

        if self.target:
            self.x_train[self.target] = train_target_data
            self.x_test[self.target] = (
                test_target_data if test_target_data is not None else None
            )
