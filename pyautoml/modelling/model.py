import multiprocessing
import warnings

import xgboost as xgb
from IPython import display
from pathos.multiprocessing import Pool
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MeanShift
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              BaggingClassifier, BaggingRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso,
                                  LinearRegression, LogisticRegression, Ridge,
                                  RidgeClassifier, SGDClassifier, SGDRegressor)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from pyautoml.base import SHELL, MethodBase, technique_reason_repo
from pyautoml.modelling.default_gridsearch_params import *
from pyautoml.modelling.model_analysis import *
from pyautoml.modelling.text import *
from pyautoml.modelling.util import (_get_cv_type, _run_models_parallel,
                                     add_to_queue, run_crossvalidation,
                                     run_gridsearch)
from pyautoml.util import (_contructor_data_properties, _input_columns,
                           _set_item, _validate_model_name)

# TODO: For classification implement probability predictions


class Model(MethodBase):
    def __init__(
        self,
        step=None,
        x_train=None,
        x_test=None,
        test_split_percentage=0.2,
        split=True,
        target_field="",
        report_name=None,
    ):

        _data_properties = _contructor_data_properties(step)

        if _data_properties is None:
            super().__init__(
                x_train=x_train,
                x_test=x_test,
                test_split_percentage=test_split_percentage,
                split=split,
                target_field=target_field,
                target_mapping=None,
                report_name=report_name,
            )
        else:
            super().__init__(
                x_train=_data_properties.x_train,
                x_test=_data_properties.x_test,
                test_split_percentage=test_split_percentage,
                split=_data_properties.split,
                target_field=_data_properties.target_field,
                target_mapping=_data_properties.target_mapping,
                report_name=_data_properties.report_name,
            )

        if self._data_properties.report is not None:
            self.report.write_header("Modelling")

        self._train_result_data = self._data_properties.x_train.copy()
        self._test_result_data = (
            self._data_properties.x_test.copy()
            if self._data_properties.x_test is not None
            else None
        )

        if self._data_properties.target_field:
            if split:
                if isinstance(step, Model):
                    self._y_train = step._y_train
                    self._y_test = step._y_test
                    self._data_properties.x_train = step._data_properties.x_train
                    self._data_properties.x_test = step._data_properties.x_test
                else:
                    self._y_train = self._data_properties.x_train[
                        self._data_properties.target_field
                    ]
                    self._y_test = self._data_properties.x_test[
                        self._data_properties.target_field
                    ]
                    self._data_properties.x_train = self._data_properties.x_train.drop(
                        [self._data_properties.target_field], axis=1
                    )
                    self._data_properties.x_test = self._data_properties.x_test.drop(
                        [self._data_properties.target_field], axis=1
                    )
            else:
                if isinstance(step, Model):
                    self._y_train = step._y_train
                    self._data_properties.x_train = step._data_properties.x_train
                else:
                    self._y_train = self._data_properties.x_train[
                        self._data_properties.target_field
                    ]
                    self._data_properties.x_train = self._data_properties.x_train.drop(
                        [self._data_properties.target_field], axis=1
                    )

        if isinstance(step, Model):
            self._models = step._models
            self._queued_models = step._queued_models
        else:
            self._models = {}
            self._queued_models = {}

    def __getitem__(self, key):

        if key in self._models:
            return self._models[key]

        return super().__getitem__(key)

    def __getattr__(self, key):

        # For when doing multi processing when pickle is reconstructing the object
        if key in {"__getstate__", "__setstate__"}:
            return object.__getattr__(self, key)

        if key in self._models:
            return self._models[key]

        try:
            if not self._data_properties.split:
                return self._train_result_data[key]
            else:
                return self._test_result_data[key]

        except Exception as e:
            raise AttributeError(e)

    def __setattr__(self, key, value):

        if key not in self.__dict__:  # any normal attributes are handled normally
            dict.__setattr__(self, key, value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):

        if key in self.__dict__:
            dict.__setitem__(self.__dict__, key, value)
        else:
            if not self._data_properties.split:
                self._train_result_data[key] = value

                return self._train_result_data.head()
            else:
                x_train_length = self._data_properties.x_train.shape[0]
                x_test_length = self._data_properties.x_test.shape[0]

                if isinstance(value, list):
                    ## If the number of entries in the list does not match the number of rows in the training or testing
                    ## set raise a value error
                    if len(value) != x_train_length and len(value) != x_test_length:
                        raise ValueError(
                            "Length of list: {} does not equal the number rows as the training set or test set.".format(
                                str(len(value))
                            )
                        )

                    self._train_result_data, self._test_result_data = _set_item(
                        self._train_result_data,
                        self._test_result_data,
                        key,
                        value,
                        x_train_length,
                        x_test_length,
                    )

                elif isinstance(value, tuple):
                    for data in value:
                        if len(data) != x_train_length and len(data) != x_test_length:
                            raise ValueError(
                                "Length of list: {} does not equal the number rows as the training set or test set.".format(
                                    str(len(data))
                                )
                            )

                        self._train_result_data, self._test_result_data = _set_item(
                            self._train_result_data,
                            self._test_result_data,
                            key,
                            data,
                            x_train_length,
                            x_test_length,
                        )

                else:
                    self._train_result_data[key] = value
                    self._test_result_data[key] = value

                return self._test_result_data.head()

    def __repr__(self):

        if SHELL == "ZMQInteractiveShell":

            display(self._train_result_data.head())  # Hack for jupyter notebooks

            return ""
        else:
            return str(self._train_result_data.head())

    @property
    def y_train(self):
        """
        Property function for the training target data.
        """

        return self._y_train

    @y_train.setter
    def y_train(self, value):
        """
        Setter function for the training target data.
        """

        self._y_train = value

    @property
    def y_test(self):
        """
        Property function for the test target data.
        """

        try:
            return self._y_test
        except Exception as e:
            return None

    @y_test.setter
    def y_test(self, value):
        """
        Setter for the test target data.
        """

        self._y_test = value

    @property
    def x_train_results(self):
        """
        Property function for the training results data.
        """

        return self._train_result_data

    @x_train_results.setter
    def x_train_results(self, value):
        """
        Setter function for the training results data.
        """

        self._train_result_data = value

    @property
    def x_test_results(self):
        """
        Property function for the test results data.
        """

        return self._test_result_data

    @x_test_results.setter
    def x_test_results(self, value):
        """
        Setter for the test target data.
        """

        self._test_result_data = value

    def run_models(self, method="parallel"):
        """
        Runs all queued models.

        The models can either be run one after the other ('series') or at the same time in parallel.

        Parameters
        ----------
        method : str, optional
            How to run models, can either be in 'series' or in 'parallel', by default 'parallel'
        """

        num_ran_models = len(self._models)

        if method == "parallel":
            _run_models_parallel(self)
        elif method == "series":
            for model in self._queued_models:
                self._queued_models[model]()
        else:
            raise ValueError(
                'Invalid run method, accepted run methods are either "parallel" or "series".'
            )

        if len(self._models) == (num_ran_models + len(self._queued_models)):
            self._queued_models = {}

    def list_models(self):
        """
        Prints out all queued and ran models.
        """

        print("######## QUEUED MODELS ########")
        if self._queued_models:
            for key in self._queued_models:
                print(key)
        else:
            print("No queued models.")

        print()

        print("######### RAN MODELS ##########")
        if self._models:
            for key in self._models:
                print(key)
        else:
            print("No ran models.")

    def delete_model(self, name):
        """
        Deletes a model, specified by it's name - can be viewed by calling list_models.

        Will look in both queued and ran models and delete where it's found.

        Parameters
        ----------
        name : str
            Name of the model
        """

        if name in self._queued_models:
            del self._queued_models[name]
        elif name in self._models:
            del self._models[name]
        else:
            raise ValueError("Model {} does not exist".format(name))

        self.list_models()

    def compare_models(self):
        """
        Compare different models across every known metric for that model.
        
        Returns
        -------
        Dataframe
            Dataframe of every model and metrics associated for that model
        """
        results = []
        for model in self._models:
            results.append(self._models[model].metrics())

        results_table = pd.concat(results, axis=1, join="inner")

        def _highlight_optimal(x):

            if "loss" in x.name.lower():
                is_min = x == x.min()
                return ["background-color: green" if v else "" for v in is_min]
            else:
                is_max = x == x.max()
                return ["background-color: green" if v else "" for v in is_max]

        results_table = results_table.style.apply(_highlight_optimal, axis=1)

        return results_table

    ################### TEXT MODELS ########################

    # TODO: Abstract these functions out to a more general template

    @add_to_queue
    def summarize_gensim(
        self,
        *list_args,
        list_of_cols=[],
        new_col_name="_summarized",
        model_name="model_summarize_gensim",
        run=False,
        **summarizer_kwargs
    ):
        """
        Summarize bodies of text using Gensim's Text Rank algorithm. Note that it uses a Text Rank variant as stated here:
        https://radimrehurek.com/gensim/summarization/summariser.html

        The output summary will consist of the most representative sentences and will be returned as a string, divided by newlines.
        
        Parameters
        ----------
        list_of_cols : list, optional
            Column name(s) of text data that you want to summarize

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `_extracted_keywords`

        model_name : str, optional
            Name for this model, default to `model_summarize_gensim`

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        ratio : float, optional
            Number between 0 and 1 that determines the proportion of the number of sentences of the original text to be chosen for the summary.

        word_count : int or None, optional
            Determines how many words will the output contain. If both parameters are provided, the ratio will be ignored.

        split : bool, optional
            If True, list of sentences will be returned. Otherwise joined strings will be returned.

        Returns
        -------
        TextModel
            Resulting model
        """

        if not _validate_model_name(self, model_name):
            raise AttributeError("Invalid model name. Please choose another one.")

        report_info = technique_reason_repo["model"]["text"]["textrank_summarizer"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        self._train_result_data, self._test_result_data = gensim_textrank_summarizer(
            x_train=self._data_properties.x_train,
            x_test=self._data_properties.x_test,
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
            **summarizer_kwargs
        )

        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = TextModel(self, model_name)

        return self._models[model_name]

    @add_to_queue
    def extract_keywords_gensim(
        self,
        *list_args,
        list_of_cols=[],
        new_col_name="_extracted_keywords",
        model_name="model_extracted_keywords_gensim",
        run=False,
        **keyword_kwargs
    ):
        """
        Extracts keywords using Gensim's implementation of the Text Rank algorithm. 

        Get most ranked words of provided text and/or its combinations.
        
        Parameters
        ----------
        list_of_cols : list, optional
            Column name(s) of text data that you want to summarize

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `_extracted_keywords`

        model_name : str, optional
            Name for this model, default to `model_extract_keywords_gensim`

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        ratio : float, optional
            Number between 0 and 1 that determines the proportion of the number of sentences of the original text to be chosen for the summary.

        words : int, optional
            Number of returned words.

        split : bool, optional
            If True, list of sentences will be returned. Otherwise joined strings will be returned.

        scores : bool, optional
            Whether score of keyword.

        pos_filter : tuple, optional
            Part of speech filters.

        lemmatize : bool, optional 
            If True - lemmatize words.

        deacc : bool, optional
            If True - remove accentuation.
        
        Returns
        -------
        TextModel
            Resulting model
        """

        if not _validate_model_name(self, model_name):
            raise AttributeError("Invalid model name. Please choose another one.")

        report_info = technique_reason_repo["model"]["text"]["textrank_keywords"]
        list_of_cols = _input_columns(list_args, list_of_cols)

        self._train_result_data, self._test_result_data = gensim_textrank_keywords(
            x_train=self._data_properties.x_train,
            x_test=self._data_properties.x_test,
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
            **keyword_kwargs
        )

        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = TextModel(self, model_name)

        return self._models[model_name]

    @add_to_queue
    def word2vec(
        self,
        col_name,
        prep=False,
        model_name="w2v",
        run=False,
        **kwargs
    ):
        """
        The underlying assumption of Word2Vec is that two words sharing similar contexts also share a similar meaning and consequently a similar vector representation from the model.
        For instance: "dog", "puppy" and "pup" are often used in similar situations, with similar surrounding words like "good", "fluffy" or "cute", and according to Word2Vec they will therefore share a similar vector representation.

        From this assumption, Word2Vec can be used to find out the relations between words in a dataset, compute the similarity between them, or use the vector representation of those words as input for other applications such as text classification or clustering.

        For more information on word2vec, you can view it here https://radimrehurek.com/gensim/models/word2vec.html.
        
        Parameters
        ----------
        col_name : str, optional
            Column name of text data that you want to summarize

        prep : bool, optional
            True to prep the data. Use when passing in raw text data.
            False if passing in text that is already prepped.
            By default False

        model_name : str, optional
            Name for this model, default to `model_extract_keywords_gensim`

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        size : int, optional
            Dimensionality of the word vectors.

        window : int, optional
            Maximum distance between the current and predicted word within a sentence.

        min_count : int, optional
            Ignores all words with total frequency lower than this.

        workers int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).

        sg : {0, 1}, optional
            Training algorithm: 1 for skip-gram; otherwise CBOW.

        hs : {0, 1}, optional
            If 1, hierarchical softmax will be used for model training.
            If 0, and negative is non-zero, negative sampling will be used.

        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.

        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution.
            A value of 1.0 samples exactly in proportion to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more than high-frequency words.
            The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that other values may perform better for recommendation applications.

        cbow_mean : {0, 1}, optional 
            If 0, use the sum of the context word vectors.
            If 1, use the mean, only applies when cbow is used.

        alpha : float, optional
            The initial learning rate.

        min_alpha : float, optional
            Learning rate will linearly drop to min_alpha as training progresses.

        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones.
            Every 10 million word types need about 1GB of RAM. Set to None for no limit.

        max_final_vocab : int, optional
            Limits the vocab to a target vocab size by automatically picking a matching min_count.
            If the specified min_count is more than the calculated min_count, the specified min_count will be used.
            Set to None if not required.

        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).

        hashfxn : function, optional
            Hash function to use to randomly initialize weights, for increased training reproducibility.

        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).

        iter : int, optional
            Number of iterations (epochs) over the corpus.

        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count). Can be None (min_count will be used, look to keep_vocab_item()), or a callable that accepts parameters (word, count, min_count) and returns either gensim.utils.RULE_DISCARD, gensim.utils.RULE_KEEP or gensim.utils.RULE_DEFAULT. The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part of the model.

            The input parameters are of the following types:

                    word (str) - the word we are examining

                    count (int) - the word’s frequency count in the corpus

                    min_count (int) - the minimum count threshold.

        sorted_vocab : {0, 1}, optional
            If 1, sort the vocabulary by descending frequency before assigning word indexes.
            See sort_vocab().

        batch_words : int, optional
            Target size (in words) for batches of examples passed to worker threads (and thus cython routines).
            (Larger batches will be passed if individual texts are longer than 10000 words, but the standard cython code truncates to that maximum.)

        compute_loss : bool, optional
            If True, computes and stores loss value which can be retrieved using get_latest_training_loss().

        Returns
        -------
        TextModel
            Resulting model
        """

        if not _validate_model_name(self, model_name):
            raise AttributeError("Invalid model name. Please choose another one.")

        report_info = technique_reason_repo["model"]["text"]["word2vec"]

        w2v_model = gensim_word2vec(
            x_train=self._data_properties.x_train,
            x_test=self._data_properties.x_test,
            prep=prep,
            col_name=col_name,
            **kwargs
        )
        
        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = TextModel(self, model_name)

        return self._models[model_name]

    @add_to_queue
    def doc2vec(
        self,
        col_name,
        prep=False,
        model_name="d2v",
        run=False,
        **kwargs
    ):
        """
        The underlying assumption of Word2Vec is that two words sharing similar contexts also share a similar meaning and consequently a similar vector representation from the model.
        For instance: "dog", "puppy" and "pup" are often used in similar situations, with similar surrounding words like "good", "fluffy" or "cute", and according to Word2Vec they will therefore share a similar vector representation.

        From this assumption, Word2Vec can be used to find out the relations between words in a dataset, compute the similarity between them, or use the vector representation of those words as input for other applications such as text classification or clustering.

        For more information on word2vec, you can view it here https://radimrehurek.com/gensim/models/word2vec.html.
        
        Parameters
        ----------
        col_name : str, optional
            Column name of text data that you want to summarize

        prep : bool, optional
            True to prep the data. Use when passing in raw text data.
            False if passing in text that is already prepped.
            By default False

        model_name : str, optional
            Name for this model, default to `model_extract_keywords_gensim`

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        dm : {1,0}, optional
            Defines the training algorithm.
            If dm=1, ‘distributed memory’ (PV-DM) is used.
            Otherwise, distributed bag of words (PV-DBOW) is employed.

        vector_size : int, optional
            Dimensionality of the feature vectors.

        window : int, optional
            The maximum distance between the current and predicted word within a sentence.

        alpha : float, optional
            The initial learning rate.

        min_alpha : float, optional
            Learning rate will linearly drop to min_alpha as training progresses.

        min_count : int, optional
            Ignores all words with total frequency lower than this.

        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; 
            if there are more unique words than this, then prune the infrequent ones.
            Every 10 million word types need about 1GB of RAM.
            Set to None for no limit.

        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).

        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).

        epochs : int, optional
            Number of iterations (epochs) over the corpus.

        hs : {1,0}, optional
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and negative is non-zero, negative sampling will be used.

        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.

        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution.
            A value of 1.0 samples exactly in proportion to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more than high-frequency words.
            The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that other values may perform better for recommendation applications.

        dm_mean : {1,0}, optional
            If 0 , use the sum of the context word vectors.
            If 1, use the mean.
            Only applies when dm is used in non-concatenative mode.

        dm_concat : {1,0}, optional
            If 1, use concatenation of context vectors rather than sum/average;
            Note concatenation results in a much-larger model, as the input is no longer the size of one (sampled or arithmetically combined) word vector,
            but the size of the tag(s) and all words in the context strung together.

        dm_tag_count : int, optional
            Expected constant number of document tags per document, when using dm_concat mode.

        dbow_words : {1,0}, optional
            If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training;
            If 0, only trains doc-vectors (faster).

        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count). Can be None (min_count will be used, look to keep_vocab_item()), or a callable that accepts parameters (word, count, min_count) and returns either gensim.utils.RULE_DISCARD, gensim.utils.RULE_KEEP or gensim.utils.RULE_DEFAULT. The rule, if given, is only used to prune vocabulary during current method call and is not stored as part of the model.

            The input parameters are of the following types:

                    word (str) - the word we are examining

                    count (int) - the word’s frequency count in the corpus

                    min_count (int) - the minimum count threshold.

        Returns
        -------
        TextModel
            Resulting model
        """

        if not _validate_model_name(self, model_name):
            raise AttributeError("Invalid model name. Please choose another one.")

        report_info = technique_reason_repo["model"]["text"]["doc2vec"]

        d2v_model = gensim_doc2vec(
            x_train=self._data_properties.x_train,
            x_test=self._data_properties.x_test,
            prep=prep,
            col_name=col_name,
            **kwargs
        )
        
        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = TextModel(self, model_name)

        return self._models[model_name]

    ################### UNSUPERVISED MODELS ########################

    @add_to_queue
    def kmeans(
        self,
        cv=None,
        gridsearch=None,
        score="homogenity_score",
        learning_curve=False,
        model_name="kmeans",
        new_col_name="kmeans_clusters",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.

        The objective of K-means is simple: group similar data points together and discover underlying patterns.
        To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset.

        In other words, the K-means algorithm identifies k number of centroids,
        and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.

        For a list of all possible options for K Means clustering please visit: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 
        
        Possible scoring metrics:
            - ‘adjusted_mutual_info_score’ 	
            - ‘adjusted_rand_score’ 	 
            - ‘completeness_score’ 	 
            - ‘fowlkes_mallows_score’ 	 
            - ‘homogeneity_score’ 	 
            - ‘mutual_info_score’ 	 
            - ‘normalized_mutual_info_score’ 	 
            - ‘v_measure_score’

        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'homogenity_score'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "kmeans"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "kmeans_clusters"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        n_clusters : int, optional, default: 8
            The number of clusters to form as well as the number of centroids to generate.

        init : {‘k-means++’, ‘random’ or an ndarray}
            Method for initialization, defaults to ‘k-means++’:
                ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.

                ‘random’: choose k observations (rows) at random from data for the initial centroids.
            If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

        n_init : int, default: 10
            Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

        max_iter : int, default: 300
            Maximum number of iterations of the k-means algorithm for a single run.

        random_state : int, RandomState instance or None (default)
            Determines random number generation for centroid initialization. Use an int to make the randomness deterministic. See Glossary.

        algorithm : “auto”, “full” or “elkan”, default=”auto”
            K-means algorithm to use.
            The classical EM-style algorithm is “full”. The “elkan” variation is more efficient by using the triangle inequality, but currently doesn’t support sparse data. 
            “auto” chooses “elkan” for dense data and “full” for sparse data.
                    
        Returns
        -------
        UnsupervisedModel
            UnsupervisedModel object to view results and further analysis
        """

        report_info = technique_reason_repo["model"]["unsupervised"]["kmeans"]
        random_state = kwargs.pop("random_state", 42)

        model = KMeans(random_state=random_state, **kwargs)

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            **kwargs
        )

        return model

    @add_to_queue
    def dbscan(
        self,
        cv=None,
        gridsearch=None,
        score="homogenity_score",
        learning_curve=False,
        model_name="dbscan",
        new_col_name="dbscan_clusters",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Based on a set of points (let’s think in a bidimensional space as exemplified in the figure), 
        DBSCAN groups together points that are close to each other based on a distance measurement (usually Euclidean distance) and a minimum number of points.
        It also marks as outliers the points that are in low-density regions.

        The DBSCAN algorithm should be used to find associations and structures in data that are hard to find manually but that can be relevant and useful to find patterns and predict trends.
        
        For a list of all possible options for DBSCAN please visit: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics:
            - ‘adjusted_mutual_info_score’ 	
            - ‘adjusted_rand_score’ 	 
            - ‘completeness_score’ 	 
            - ‘fowlkes_mallows_score’ 	 
            - ‘homogeneity_score’ 	 
            - ‘mutual_info_score’ 	 
            - ‘normalized_mutual_info_score’ 	 
            - ‘v_measure_score’

        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'homogenity_score'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "dbscan"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "dbscan_clusters"

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        eps : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            This is not a maximum bound on the distances of points within a cluster.
            This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.

        min_samples : int, optional
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.

        metric : string, or callable
            The metric to use when calculating distance between instances in a feature array.
            If metric is a string or callable, it must be one of the options allowed by sklearn.
            If metric is “precomputed”, X is assumed to be a distance matrix and must be square.
            X may be a sparse matrix, in which case only “nonzero” elements may be considered neighbors for DBSCAN.

        p : float, optional
            The power of the Minkowski metric to be used to calculate distance between points.

        n_jobs : int or None, optional (default=None)
            The number of parallel jobs to run.
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
            See Glossary for more details.
        
        Returns
        -------
        UnsupervisedModel
            UnsupervisedModel object to view results and further analysis
        """

        report_info = technique_reason_repo["model"]["unsupervised"]["dbscan"]

        model = DBSCAN(**kwargs)

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            **kwargs
        )

        return model

    @add_to_queue
    def isolation_forest(
        self,
        cv=None,
        gridsearch=None,
        score="homogenity_score",
        learning_curve=False,
        model_name="iso_forest",
        new_col_name="iso_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Isolation Forest Algorithm

        Return the anomaly score of each sample using the IsolationForest algorithm

        The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

        For more Isolation Forest info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics:
            - ‘adjusted_mutual_info_score’ 	
            - ‘adjusted_rand_score’ 	 
            - ‘completeness_score’ 	 
            - ‘fowlkes_mallows_score’ 	 
            - ‘homogeneity_score’ 	 
            - ‘mutual_info_score’ 	 
            - ‘normalized_mutual_info_score’ 	 
            - ‘v_measure_score’
        
        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "iso_forest"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "iso_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        n_estimators : int, optional (default=100)
            The number of base estimators in the ensemble.

        max_samples : int or float, optional (default=”auto”)
            The number of samples to draw from X to train each base estimator.

                    If int, then draw max_samples samples.
                    If float, then draw max_samples * X.shape[0] samples.
                    If “auto”, then max_samples=min(256, n_samples).

            If max_samples is larger than the number of samples provided, all samples will be used for all trees (no sampling).

        contamination : float in (0., 0.5), optional (default=0.1)
            The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
            Used when fitting to define the threshold on the decision function.
            If ‘auto’, the decision function threshold is determined as in the original paper.

        max_features : int or float, optional (default=1.0)
            The number of features to draw from X to train each base estimator.

                    If int, then draw max_features features.
                    If float, then draw max_features * X.shape[1] features.

        bootstrap : boolean, optional (default=False)
            If True, individual trees are fit on random subsets of the training data sampled with replacement.
            If False, sampling without replacement is performed.

        Returns
        -------
        UnsupervisedModel
            UnsupervisedModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["unsupervised"]["iso_forest"]

        model = IsolationForest(random_state=random_state, **kwargs)

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def oneclass_svm(
        self,
        cv=None,
        gridsearch=None,
        score="homogenity_score",
        learning_curve=False,
        model_name="ocsvm",
        new_col_name="ocsvm_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a One Class SVM model.

        Unsupervised Outlier Detection.

        For more Support Vector info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics:
            - ‘adjusted_mutual_info_score’ 	
            - ‘adjusted_rand_score’ 	 
            - ‘completeness_score’ 	 
            - ‘fowlkes_mallows_score’ 	 
            - ‘homogeneity_score’ 	 
            - ‘mutual_info_score’ 	 
            - ‘normalized_mutual_info_score’ 	 
            - ‘v_measure_score’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "ocsvm"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "ocsvm_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False    	

        kernel : string, optional (default=’rbf’)
            Specifies the kernel type to be used in the algorithm.
            It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
            If none is given, ‘rbf’ will be used.
            If a callable is given it is used to precompute the kernel matrix.

        degree : int, optional (default=3)
            Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

        gamma : float, optional (default=’auto’)
            Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            Current default is ‘auto’ which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma.

        coef0 : float, optional (default=0.0)
            Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

        tol : float, optional
            Tolerance for stopping criterion.

        nu : float, optional
            An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
            Should be in the interval (0, 1]. By default 0.5 will be taken.

        shrinking : boolean, optional
            Whether to use the shrinking heuristic.

        cache_size : float, optional
            Specify the size of the kernel cache (in MB).
        
        max_iter : int, optional (default=-1)
            Hard limit on iterations within solver, or -1 for no limit.

        Returns
        -------
        UnsupervisedModel
            UnsupervisedModel object to view results and analyze results
        """

        report_info = technique_reason_repo["model"]["unsupervised"]["oneclass_cls"]

        model = OneClassSVM(**kwargs)

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            **kwargs
        )

        return model

    @add_to_queue
    def agglomerative_clustering(
        self,
        cv=None,
        gridsearch=None,
        score="homogenity_score",
        learning_curve=False,
        model_name="agglom",
        new_col_name="agglom_clusters",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Agglomerative Clustering Model

        Each data point as a single cluster at the outset and then successively merge (or agglomerate) pairs of clusters until all clusters have been merged into a single cluster that contains all data points
        
        Hierarchical clustering does not require us to specify the number of clusters and we can even select which number of clusters looks best since we are building a tree.
        
        Additionally, the algorithm is not sensitive to the choice of distance metric; all of them tend to work equally well whereas with other clustering algorithms, 
        the choice of distance metric is critical. 
        
        A particularly good use case of hierarchical clustering methods is when the underlying data has a hierarchical structure and you want to recover the hierarchy;
        other clustering algorithms can’t do this.
        
        These advantages of hierarchical clustering come at the cost of lower efficiency, as it has a time complexity of O(n³), unlike the linear complexity of K-Means and GMM.

        For a list of all possible options for Agglomerative clustering please visit: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 
        
        Possible scoring metrics:
            - ‘adjusted_mutual_info_score’ 	
            - ‘adjusted_rand_score’ 	 
            - ‘completeness_score’ 	 
            - ‘fowlkes_mallows_score’ 	 
            - ‘homogeneity_score’ 	 
            - ‘mutual_info_score’ 	 
            - ‘normalized_mutual_info_score’ 	 
            - ‘v_measure_score’

        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'homogenity_score'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "agglom"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "agglom_clusters"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        n_clusters : int or None, optional (default=2)
            The number of clusters to find.
            It must be None if distance_threshold is not None.

        affinity : string or callable, default: “euclidean”
            Metric used to compute the linkage.
            Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
            If linkage is “ward”, only “euclidean” is accepted.
            If “precomputed”, a distance matrix (instead of a similarity matrix) is needed as input for the fit method.

        compute_full_tree : bool or ‘auto’ (optional)
            Stop early the construction of the tree at n_clusters.
            This is useful to decrease computation time if the number of clusters is not small compared to the number of samples.
            This option is useful only when specifying a connectivity matrix.
            Note also that when varying the number of clusters and using caching, it may be advantageous to compute the full tree.
            It must be True if distance_threshold is not None.

        linkage : {“ward”, “complete”, “average”, “single”}, optional (default=”ward”)
            Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.

                'ward' minimizes the variance of the clusters being merged.
                'average' uses the average of the distances of each observation of the two sets.
                'complete' or maximum linkage uses the maximum distances between all observations of the two sets.
                'single' uses the minimum of the distances between all observations of the two sets.

        distance_threshold : float, optional (default=None)
            The linkage distance threshold above which, clusters will not be merged.
            If not None, n_clusters must be None and compute_full_tree must be True.
                    
        Returns
        -------
        UnsupervisedModel
            UnsupervisedModel object to view results and further analysis
        """

        report_info = technique_reason_repo["model"]["unsupervised"]["agglom"]

        model = AgglomerativeClustering(**kwargs)

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            **kwargs
        )

        return model

    @add_to_queue
    def mean_shift(
        self,
        cv=None,
        gridsearch=None,
        score="homogenity_score",
        learning_curve=False,
        model_name="mshift",
        new_col_name="mshift_clusters",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Mean Shift clustering algorithm.

        Mean shift clustering aims to discover “blobs” in a smooth density of samples.

        It is a centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region.

        These candidates are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids.

        For more info on Mean Shift clustering please visit: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 
        
        Possible scoring metrics:
            - ‘adjusted_mutual_info_score’ 	
            - ‘adjusted_rand_score’ 	 
            - ‘completeness_score’ 	 
            - ‘fowlkes_mallows_score’ 	 
            - ‘homogeneity_score’ 	 
            - ‘mutual_info_score’ 	 
            - ‘normalized_mutual_info_score’ 	 
            - ‘v_measure_score’

        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'homogenity_score'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "mshift"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "mshift_clusters"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        bandwidth : float, optional
            Bandwidth used in the RBF kernel.

            If not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth; see the documentation for that function for hints on scalability (see also the Notes, below).

        seeds : array, shape=[n_samples, n_features], optional
            Seeds used to initialize kernels.
            If not set, the seeds are calculated by clustering.get_bin_seeds with bandwidth as the grid size and default values for other parameters.

        bin_seeding : boolean, optional
            If true, initial kernel locations are not locations of all points, but rather the location of the discretized version of points, where points are binned onto a grid whose coarseness corresponds to the bandwidth.
            Setting this option to True will speed up the algorithm because fewer seeds will be initialized.
            default value: False Ignored if seeds argument is not None.        
            
        min_bin_freq : int, optional
            To speed up the algorithm, accept only those bins with at least min_bin_freq points as seeds.
            If not defined, set to 1.

        cluster_all : boolean, default True
            If true, then all points are clustered, even those orphans that are not within any kernel. Orphans are assigned to the nearest kernel.
            If false, then orphans are given cluster label -1.
                    
        Returns
        -------
        UnsupervisedModel
            UnsupervisedModel object to view results and further analysis
        """

        report_info = technique_reason_repo["model"]["unsupervised"]["ms"]

        model = MeanShift(**kwargs)

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            **kwargs
        )

        return model

    @add_to_queue
    def gaussian_mixture_clustering(
        self,
        cv=None,
        gridsearch=None,
        score="homogenity_score",
        learning_curve=False,
        model_name="gm_cluster",
        new_col_name="gm_clusters",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a GaussianMixture algorithm that implements the expectation-maximization algorithm for fitting mixture
        of Gaussian models.

        A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

        There are 2 key advantages to using GMMs.
        
        Firstly GMMs are a lot more flexible in terms of cluster covariance than K-Means; due to the standard deviation parameter, the clusters can take on any ellipse shape, rather than being restricted to circles.
        
        K-Means is actually a special case of GMM in which each cluster’s covariance along all dimensions approaches 0.
        Secondly, since GMMs use probabilities, they can have multiple clusters per data point.
        
        So if a data point is in the middle of two overlapping clusters, we can simply define its class by saying it belongs X-percent to class 1 and Y-percent to class 2. I.e GMMs support mixed membership.

        For more information on Gaussian Mixture algorithms please visit: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 
        
        Possible scoring metrics:
            - ‘adjusted_mutual_info_score’ 	
            - ‘adjusted_rand_score’ 	 
            - ‘completeness_score’ 	 
            - ‘fowlkes_mallows_score’ 	 
            - ‘homogeneity_score’ 	 
            - ‘mutual_info_score’ 	 
            - ‘normalized_mutual_info_score’ 	 
            - ‘v_measure_score’

        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'homogenity_score'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "gm_cluster"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "gm_clusters"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        n_components : int, defaults to 1.
            The number of mixture components/ number of unique y_train values.

        covariance_type : {‘full’ (default), ‘tied’, ‘diag’, ‘spherical’}
            String describing the type of covariance parameters to use. Must be one of:

            ‘full’
                each component has its own general covariance matrix

            ‘tied’
                all components share the same general covariance matrix

            ‘diag’
                each component has its own diagonal covariance matrix

            ‘spherical’
                each component has its own single variance

        tol : float, defaults to 1e-3.
            The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.

        reg_covar : float, defaults to 1e-6.
            Non-negative regularization added to the diagonal of covariance.
            Allows to assure that the covariance matrices are all positive.

        max_iter : int, defaults to 100.
            The number of EM iterations to perform.

        n_init : int, defaults to 1.
            The number of initializations to perform. The best results are kept.

        init_params : {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
            The method used to initialize the weights, the means and the precisions. Must be one of:

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

        weights_init : array-like, shape (n_components, ), optional
            The user-provided initial weights
            If it None, weights are initialized using the init_params method.
            Defaults to None. 

        means_init : array-like, shape (n_components, n_features), optional
            The user-provided initial means
            If it None, means are initialized using the init_params method.
            Defaults to None

        precisions_init : array-like, optional.
            The user-provided initial precisions (inverse of the covariance matrices), defaults to None. If it None, precisions are initialized using the ‘init_params’ method. The shape depends on ‘covariance_type’:

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
            
        Returns
        -------
        UnsupervisedModel
            UnsupervisedModel object to view results and further analysis
        """

        report_info = technique_reason_repo["model"]["unsupervised"]["em_gmm"]
        random_state = kwargs.pop("random_state", 42)

        model = GaussianMixture(random_state=random_state, **kwargs)

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            **kwargs
        )

        return model

    ################### CLASSIFICATION MODELS ########################

    # NOTE: This entire process may need to be reworked.
    @add_to_queue
    def logistic_regression(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="log_reg",
        new_col_name="log_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a logistic regression model.

        For more Logistic Regression info, you can view them here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "log_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "log_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        penalty : str, ‘l1’, ‘l2’, ‘elasticnet’ or ‘none’, optional (default=’l2’)
            Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. 
            ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no regularization is applied.

        tol : float, optional (default=1e-4)
            Tolerance for stopping criteria.

        C : float, optional (default=1.0)
            Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

        class_weight : dict or ‘balanced’, optional (default=None)
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
            Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
        
        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        solver = kwargs.pop("solver", "lbfgs")
        report_info = technique_reason_repo["model"]["classification"]["logreg"]

        model = LogisticRegression(solver=solver, random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def ridge_classification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="ridge_cls",
        new_col_name="ridge_cls_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Ridge Classification model.

        For more Ridge Regression parameters, you can view them here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier        

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "ridge_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "ridge_cls_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        alpha : float
            Regularization strength; must be a positive float.
            Regularization improves the conditioning of the problem and reduces the variance of the estimates.
            Larger values specify stronger regularization.
            Alpha corresponds to C^-1 in other linear models such as LogisticRegression or LinearSVC.

        fit_intercept : boolean
            Whether to calculate the intercept for this model.
            If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered).

        normalize : boolean, optional, default False
            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.

        tol : float, optional (default=1e-4)
            Tolerance for stopping criteria.

        class_weight : dict or ‘balanced’, optional (default=None)
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
            Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["classification"]["ridge_cls"]

        model = RidgeClassifier(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def sgd_classification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="sgd_cls",
        new_col_name="sgd_cls_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Linear classifier (SVM, logistic regression, a.o.) with SGD training.

        For more info please view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'
        
        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "sgd_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "sgd_cls_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False
        
        loss : str, default: ‘hinge’
            The loss function to be used. Defaults to ‘hinge’, which gives a linear SVM.
            The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a regression loss: ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.

            The ‘log’ loss gives logistic regression, a probabilistic classifier. 
            ‘modified_huber’ is another smooth loss that brings tolerance to outliers as well as probability estimates. 
            ‘squared_hinge’ is like hinge but is quadratically penalized. 
            ‘perceptron’ is the linear loss used by the perceptron algorithm.
            The other losses are designed for regression but can be useful in classification as well; see SGDRegressor for a description.

        penalty : str, ‘none’, ‘l2’, ‘l1’, or ‘elasticnet’
            The penalty (aka regularization term) to be used.
            Defaults to ‘l2’ which is the standard regularizer for linear SVM models.
            ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.
        
        alpha : float
            Constant that multiplies the regularization term. Defaults to 0.0001 Also used to compute learning_rate when set to ‘optimal’.

        l1_ratio : float
            The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15.

        fit_intercept : bool
            Whether the intercept should be estimated or not. If False, the data is assumed to be already centered. Defaults to True.

        max_iter : int, optional (default=1000)
            The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit.

        tol : float or None, optional (default=1e-3)
            The stopping criterion. If it is not None, the iterations will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs.

        shuffle : bool, optional
            Whether or not the training data should be shuffled after each epoch. Defaults to True.

        epsilon : float
            Epsilon in the epsilon-insensitive loss functions; only if loss is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’. For ‘huber’, determines the threshold at which it becomes less important to get the prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold.

        learning_rate : string, optional

            The learning rate schedule:

            ‘constant’:

                eta = eta0
            ‘optimal’: [default]

                eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
            ‘invscaling’:

                eta = eta0 / pow(t, power_t)
            ‘adaptive’:

                eta = eta0, as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True, the current learning rate is divided by 5.

        eta0 : double
            The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.0 as eta0 is not used by the default schedule ‘optimal’.

        power_t : double
            The exponent for inverse scaling learning rate [default 0.5].

        early_stopping : bool, default=False
            Whether to use early stopping to terminate training when validation score is not improving.
            If set to True, it will automatically set aside a stratified fraction of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.

        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True.

        n_iter_no_change : int, default=5
            Number of iterations with no improvement to wait before early stopping.

        class_weight : dict, {class_label: weight} or “balanced” or None, optional
            Preset for the class_weight fit parameter.

            Weights associated with classes. If not given, all classes are supposed to have weight one.

            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))

        average : bool or int, optional
            When set to True, computes the averaged SGD weights and stores the result in the coef_ attribute.
            If set to an int greater than 1, averaging will begin once the total number of samples seen reaches average. So average=10 will begin averaging after seeing 10 samples.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["classification"]["sgd_cls"]

        model = SGDClassifier(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def adaboost_classification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="ada_cls",
        new_col_name="ada_cls_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains an AdaBoost classification model.

        An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset
        but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

        For more AdaBoost info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "ada_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "ada_cls_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        base_estimator : object, optional (default=None)
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes.
            If None, then the base estimator is DecisionTreeClassifier(max_depth=1)

        n_estimators : integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.

        learning_rate : float, optional (default=1.)
            Learning rate shrinks the contribution of each classifier by learning_rate.
            There is a trade-off between learning_rate and n_estimators.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["classification"]["ada_cls"]

        model = AdaBoostClassifier(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def bagging_classification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="bag_cls",
        new_col_name="bag_cls_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Bagging classification model.

        A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
        Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

        For more Bagging Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "bag_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "bag_cls_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        base_estimator : object or None, optional (default=None)
            The base estimator to fit on random subsets of the dataset.
            If None, then the base estimator is a decision tree.

        n_estimators : int, optional (default=10)
            The number of base estimators in the ensemble.

        max_samples : int or float, optional (default=1.0)
            The number of samples to draw from X to train each base estimator.

                If int, then draw max_samples samples.
                If float, then draw max_samples * X.shape[0] samples.

        max_features : int or float, optional (default=1.0)
            The number of features to draw from X to train each base estimator.

                If int, then draw max_features features.
                If float, then draw max_features * X.shape[1] features.

        bootstrap : boolean, optional (default=True)
            Whether samples are drawn with replacement. If False, sampling without replacement is performed.

        bootstrap_features : boolean, optional (default=False)
            Whether features are drawn with replacement.

        oob_score : bool, optional (default=False)
            Whether to use out-of-bag samples to estimate the generalization error.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["classification"]["bag_cls"]

        model = BaggingClassifier(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def gradient_boosting_classification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="grad_cls",
        new_col_name="grad_cls_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Gradient Boosting classification model.

        GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.
        In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. 
        Binary classification is a special case where only a single regression tree is induced.

        For more Gradient Boosting Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier   

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "grad_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "grad_cls_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        loss : {‘deviance’, ‘exponential’}, optional (default=’deviance’)
            loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. 
            For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.
            
        learning_rate : float, optional (default=0.1)
            learning rate shrinks the contribution of each tree by learning_rate.
            There is a trade-off between learning_rate and n_estimators.

        n_estimators : int (default=100)
            The number of boosting stages to perform.
            Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.

        subsample : float, optional (default=1.0)
            The fraction of samples to be used for fitting the individual base learners.
            If smaller than 1.0 this results in Stochastic Gradient Boosting.
            Subsample interacts with the parameter n_estimators.
            Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.

        criterion : string, optional (default=”friedman_mse”)
            The function to measure the quality of a split.
            Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error.
            The default value of “friedman_mse” is generally the best as it can provide a better approximation in some cases.

        min_samples_split : int, float, optional (default=2)
            The minimum number of samples required to split an internal node:

                If int, then consider min_samples_split as the minimum number.
                If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

        min_samples_leaf : int, float, optional (default=1)
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
            This may have the effect of smoothing the model, especially in regression.

                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

        max_depth : integer, optional (default=3)
            maximum depth of the individual regression estimators.
            The maximum depth limits the number of nodes in the tree.
            Tune this parameter for best performance; the best value depends on the interaction of the input variables.

        max_features : int, float, string or None, optional (default=None)
            The number of features to consider when looking for the best split:

                If int, then consider max_features features at each split.
                If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
                If “auto”, then max_features=sqrt(n_features).
                If “sqrt”, then max_features=sqrt(n_features).
                If “log2”, then max_features=log2(n_features).
                If None, then max_features=n_features.

            Choosing max_features < n_features leads to a reduction of variance and an increase in bias.

            Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features. 

        max_leaf_nodes : int or None, optional (default=None)
            Grow trees with max_leaf_nodes in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        presort : bool or ‘auto’, optional (default=’auto’)
            Whether to presort the data to speed up the finding of best splits in fitting.
            Auto mode by default will use presorting on dense data and default to normal sorting on sparse data.
            Setting presort to true on sparse data will raise an error.

        validation_fraction : float, optional, default 0.1
            The proportion of training data to set aside as validation set for early stopping.
            Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.

        tol : float, optional, default 1e-4
            Tolerance for the early stopping.
            When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["classification"]["grad_cls"]

        model = GradientBoostingClassifier(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def random_forest_classification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="rf_cls",
        new_col_name="rf_cls_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Random Forest classification model.

        A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
        The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).

        For more Random Forest info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "rf_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "rf_cls_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False
        
        n_estimators : integer, optional (default=10)
            The number of trees in the forest.

        criterion : string, optional (default=”gini”)
            The function to measure the quality of a split.
            Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
            
            Note: this parameter is tree-specific.

        max_depth : integer or None, optional (default=None)
            The maximum depth of the tree.
            If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

        min_samples_split : int, float, optional (default=2)
            The minimum number of samples required to split an internal node:

                If int, then consider min_samples_split as the minimum number.
                If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

        min_samples_leaf : int, float, optional (default=1)
            The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.

                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

        max_features : int, float, string or None, optional (default=”auto”)
            The number of features to consider when looking for the best split:

                If int, then consider max_features features at each split.
                If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
                If “auto”, then max_features=sqrt(n_features).
                If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
                If “log2”, then max_features=log2(n_features).
                If None, then max_features=n_features.

            Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
        
        max_leaf_nodes : int or None, optional (default=None)
            Grow trees with max_leaf_nodes in best-first fashion.
            Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, optional (default=0.)
            A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

            The weighted impurity decrease equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

            where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.

            N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.

        bootstrap : boolean, optional (default=True)
            Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.

        oob_score : bool (default=False)
            Whether to use out-of-bag samples to estimate the generalization accuracy.

        class_weight : dict, list of dicts, “balanced”, “balanced_subsample” or None, optional (default=None)
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
            Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
            The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.
            For multi-output, the weights of each column of y will be multiplied.

            Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["classification"]["rf_cls"]

        model = RandomForestClassifier(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def nb_bernoulli_classification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="bern",
        new_col_name="bern_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Bernoulli Naive Bayes classification model.

        Like MultinomialNB, this classifier is suitable for discrete data.
        The difference is that while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features.

        For more Bernoulli Naive Bayes info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
        and https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes 

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default False.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "bern"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "bern_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False
        
        alpha : float, optional (default=1.0)
            Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

        binarize : float or None, optional (default=0.0)
            Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors.

        fit_prior : boolean, optional (default=True)
            Whether to learn class prior probabilities or not. If false, a uniform prior will be used.

        class_prior : array-like, size=[n_classes,], optional (default=None)
            Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        report_info = technique_reason_repo["model"]["classification"]["bern"]

        model = BernoulliNB(**kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            **kwargs
        )

        return model

    @add_to_queue
    def nb_gaussian_classification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="gauss",
        new_col_name="gauss_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Gaussian Naive Bayes classification model.

        For more Gaussian Naive Bayes info, you can view it here: https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default False.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "gauss"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "gauss_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        priors : array-like, shape (n_classes,)
            Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

        var_smoothing : float, optional (default=1e-9)
            Portion of the largest variance of all features that is added to variances for calculation stability.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        report_info = technique_reason_repo["model"]["classification"]["gauss"]

        model = GaussianNB(**kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            **kwargs
        )

        return model

    @add_to_queue
    def nb_multinomial_classification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="multi",
        new_col_name="multi_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Multinomial Naive Bayes classification model.

        The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts.
        However, in practice, fractional counts such as tf-idf may also work.

        For more Multinomial Naive Bayes info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
        and https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes 

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default False.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "multi"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "multi_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        alpha : float, optional (default=1.0)
            Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

        fit_prior : boolean, optional (default=True)
            Whether to learn class prior probabilities or not. If false, a uniform prior will be used.

        class_prior : array-like, size (n_classes,), optional (default=None)
            Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        report_info = technique_reason_repo["model"]["classification"]["multi"]

        model = MultinomialNB(**kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            **kwargs
        )

        return model

    @add_to_queue
    def decision_tree_classification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="dt_cls",
        new_col_name="dt_cls_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Decision Tree classification model.

        For more Decision Tree info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "dt_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "dt_cls_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False   
        	
        criterion : string, optional (default=”gini”)
            The function to measure the quality of a split.
            Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.

        splitter : string, optional (default=”best”)
            The strategy used to choose the split at each node.
            Supported strategies are “best” to choose the best split and “random” to choose the best random split.

        max_depth : int or None, optional (default=None)
            The maximum depth of the tree.
            If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

        min_samples_split : int, float, optional (default=2)
            The minimum number of samples required to split an internal node:

                If int, then consider min_samples_split as the minimum number.
                If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

        min_samples_leaf : int, float, optional (default=1)
            The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.

                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

        max_features : int, float, string or None, optional (default=None)
            The number of features to consider when looking for the best split:

                    If int, then consider max_features features at each split.
                    If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
                    If “auto”, then max_features=sqrt(n_features).
                    If “sqrt”, then max_features=sqrt(n_features).
                    If “log2”, then max_features=log2(n_features).
                    If None, then max_features=n_features.

            Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.

        max_leaf_nodes : int or None, optional (default=None)
            Grow a tree with max_leaf_nodes in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, optional (default=0.)
            A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

            The weighted impurity decrease equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

            where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.

            N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.

        min_impurity_split : float, (default=1e-7)
            Threshold for early stopping in tree growth.
            A node will split if its impurity is above the threshold, otherwise it is a leaf.

        class_weight : dict, list of dicts, “balanced” or None, default=None
            Weights associated with classes in the form {class_label: weight}.
            If not given, all classes are supposed to have weight one.
            For multi-output problems, a list of dicts can be provided in the same order as the columns of y.

            Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict.
            For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].

            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))

            For multi-output, the weights of each column of y will be multiplied.

            Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

        presort : bool, optional (default=False)
            Whether to presort the data to speed up the finding of best splits in fitting.
            For the default settings of a decision tree on large datasets, setting this to true may slow down the training process.
            When using either a smaller dataset or a restricted depth, this may speed up the training.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["classification"]["dt_cls"]

        model = DecisionTreeClassifier(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def linearsvc(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="linsvc",
        new_col_name="linsvc_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Linear Support Vector classification model.

        Supports multi classification.

        Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
        This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme.

        For more Support Vector info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "linsvc"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "linsvc_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False    	

        penalty : string, ‘l1’ or ‘l2’ (default=’l2’)
            Specifies the norm used in the penalization.
            The ‘l2’ penalty is the standard used in SVC.
            The ‘l1’ leads to coef_ vectors that are sparse.

        loss : string, ‘hinge’ or ‘squared_hinge’ (default=’squared_hinge’)
            Specifies the loss function.            
            ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the square of the hinge loss.

        dual : bool, (default=True)
            Select the algorithm to either solve the dual or primal optimization problem.
            Prefer dual=False when n_samples > n_features.

        tol : float, optional (default=1e-4)
            Tolerance for stopping criteria.

        C : float, optional (default=1.0)
            Penalty parameter C of the error term.

        multi_class : string, ‘ovr’ or ‘crammer_singer’ (default=’ovr’)
            Determines the multi-class strategy if y contains more than two classes.
            "ovr" trains n_classes one-vs-rest classifiers, while "crammer_singer" optimizes a joint objective over all classes.
            While crammer_singer is interesting from a theoretical perspective as it is consistent, it is seldom used in practice as it rarely leads to better accuracy and is more expensive to compute.
            If "crammer_singer" is chosen, the options loss, penalty and dual will be ignored.

        fit_intercept : boolean, optional (default=True)
            Whether to calculate the intercept for this model.
            If set to false, no intercept will be used in calculations (i.e. data is expected to be already centered).

        intercept_scaling : float, optional (default=1)
            When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance vector.
            The intercept becomes intercept_scaling * synthetic feature weight Note! the synthetic feature weight is subject to l1/l2 regularization as all other features.
            To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.

        class_weight : {dict, ‘balanced’}, optional
            Set the parameter C of class i to class_weight[i]*C for SVC.
            If not given, all classes are supposed to have weight one.
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
       
        max_iter : int, (default=1000)
            The maximum number of iterations to be run.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["classification"]["linsvc"]

        model = LinearSVC(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def svc(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="svc",
        new_col_name="svc_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a C-Support Vector classification model.

        Supports multi classification.

        The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples.
        For large datasets consider using model.linearsvc or model.sgd_classification instead

        The multiclass support is handled according to a one-vs-one scheme.

        For more Support Vector info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "linsvc_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "linsvc_cls_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False    	

        C : float, optional (default=1.0)
            Penalty parameter C of the error term.

        kernel : string, optional (default=’rbf’)
            Specifies the kernel type to be used in the algorithm.
            It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
            If none is given, ‘rbf’ will be used.
            If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).

        degree : int, optional (default=3)
            Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

        gamma : float, optional (default=’auto’)
            Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            Current default is ‘auto’ which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma.

        coef0 : float, optional (default=0.0)
            Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

        shrinking : boolean, optional (default=True)
            Whether to use the shrinking heuristic.

        probability : boolean, optional (default=False)
            Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.

        tol : float, optional (default=1e-3)
            Tolerance for stopping criterion.

        cache_size : float, optional
            Specify the size of the kernel cache (in MB).

        class_weight : {dict, ‘balanced’}, optional
            Set the parameter C of class i to class_weight[i]*C for SVC.
            If not given, all classes are supposed to have weight one.
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))

        max_iter : int, optional (default=-1)
            Hard limit on iterations within solver, or -1 for no limit.

        decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’
            Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers,
            or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).
            However, one-vs-one (‘ovo’) is always used as multi-class strategy.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["classification"]["svc"]

        model = SVC(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def xgboost_classification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        model_name="xgb_cls",
        new_col_name="xgb_cls_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains an XGBoost Classification Model.

        XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
        It implements machine learning algorithms under the Gradient Boosting framework.
        XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.
        The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.

        For more XGBoost info, you can view it here: https://xgboost.readthedocs.io/en/latest/ and
        https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst. 

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "xgb_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "xgb_cls_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False    	

        max_depth : int
            Maximum tree depth for base learners. By default 3

        learning_rate : float
            Boosting learning rate (xgb's "eta"). By default 0.1

        n_estimators : int
            Number of trees to fit. By default 100.

        objective : string or callable
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            By default binary:logistic for binary classification or multi:softprob for multiclass classifcation

        booster: string
            Specify which booster to use: gbtree, gblinear or dart. By default 'gbtree'

        tree_method: string
            Specify which tree method to use
            If this parameter is set to default, XGBoost will choose the most conservative option
            available.  It's recommended to study this option from parameters
            document. By default 'auto'

        gamma : float
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
            By default 0

        subsample : float
            Subsample ratio of the training instance.
            By default 1
        
        reg_alpha : float (xgb's alpha)
            L1 regularization term on weights. By default 0

        reg_lambda : float (xgb's lambda)
            L2 regularization term on weights. By default 1

        scale_pos_weight : float
            Balancing of positive and negative weights. By default 1

        base_score:
            The initial prediction score of all instances, global bias. By default 0

        missing : float, optional
            Value in the data which needs to be present as a missing value. If
            None, defaults to np.nan.
            By default, None

        num_parallel_tree: int
            Used for boosting random forest.
            By default 1

        importance_type: string, default "gain"
            The feature importance type for the feature_importances\\_ property:
            either "gain", "weight", "cover", "total_gain" or "total_cover".
            By default 'gain'.

        Note
        ----
        A custom objective function can be provided for the ``objective``
        parameter. In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``:

        y_true: array_like of shape [n_samples]
            The target values

        y_pred: array_like of shape [n_samples]
            The predicted values

        grad: array_like of shape [n_samples]
            The value of the gradient for each sample point.

        hess: array_like of shape [n_samples]
            The value of the second derivative for each sample point

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        objective = kwargs.pop('objective', 'binary:logistic' if len(self._y_train.unique()) == 2 else 'multi:softprob')
        report_info = technique_reason_repo["model"]["classification"]["xgb_cls"]

        model = xgb.XGBClassifier(objective=objective, random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    ################### REGRESSION MODELS ########################

    @add_to_queue
    def linear_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="lin_reg",
        new_col_name="linreg_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Linear Regression.

        For more Linear Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default ‘neg_mean_squared_error’

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "lin_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "linreg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False        	

        fit_intercept : boolean, optional, default True
            whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).

        normalize : boolean, optional, default False
            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["linreg"]

        model = LinearRegression(**kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def bayesian_ridge_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="bayridge_reg",
        new_col_name="bayridge_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Bayesian Ridge Regression model.

        For more Linear Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
        and https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression 

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default ‘neg_mean_squared_error’

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "bayridge_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "bayridge_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False
        
        n_iter : int, optional
            Maximum number of iterations. Default is 300. Should be greater than or equal to 1.

        tol : float, optional
            Stop the algorithm if w has converged. Default is 1.e-3.
            
        alpha_1 : float, optional
            Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter. Default is 1.e-6

        alpha_2 : float, optional
            Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter. Default is 1.e-6.

        lambda_1 : float, optional
            Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter. Default is 1.e-6.

        lambda_2 : float, optional
            Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter. Default is 1.e-6

        fit_intercept : boolean, optional, default True
            Whether to calculate the intercept for this model.
            The intercept is not treated as a probabilistic parameter and thus has no associated variance.
            If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).

        normalize : boolean, optional, default False
            This parameter is ignored when fit_intercept is set to False. 
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
            
        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["bay_reg"]

        model = BayesianRidge(**kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def elasticnet_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="elastic",
        new_col_name="elastic_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Elastic Net regression with combined L1 and L2 priors as regularizer.
        
        For more Linear Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet 

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 
        
        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default ‘neg_mean_squared_error’

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "elastic"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "elastic_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False   
        
        alpha : float, optional
            Constant that multiplies the penalty terms.
            Defaults to 1.0. See the notes for the exact mathematical meaning of this parameter.
            ``alpha = 0`` is equivalent to an ordinary least square, solved by the LinearRegression object.
            For numerical reasons, using alpha = 0 with the Lasso object is not advised.
            Given this, you should use the LinearRegression object.

        l1_ratio : float
            The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
            For l1_ratio = 0 the penalty is an L2 penalty.
            For l1_ratio = 1 it is an L1 penalty.
            For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

        fit_intercept : bool
            Whether the intercept should be estimated or not.
            If False, the data is assumed to be already centered.

        normalize : boolean, optional, default False
            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
            If you wish to standardize, please use sklearn.preprocessing.

        precompute : True | False | array-like
            Whether to use a precomputed Gram matrix to speed up calculations.
            The Gram matrix can also be passed as argument.
            For sparse input this option is always True to preserve sparsity.

        max_iter : int, optional
            The maximum number of iterations

        tol : float, optional
            The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol.
        
        positive : bool, optional
            When set to True, forces the coefficients to be positive.

        selection : str, default ‘cyclic’
            If set to ‘random’, a random coefficient is updated every iteration rather than looping over features sequentially by default.
            This (setting to ‘random’) often leads to significantly faster convergence especially when tol is higher than 1e-4.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["el_net"]

        model = ElasticNet(**kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def lasso_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="lasso",
        new_col_name="lasso_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Lasso Regression Model trained with L1 prior as regularizer (aka the Lasso)

        Technically the Lasso model is optimizing the same objective function as the Elastic Net with l1_ratio=1.0 (no L2 penalty).   

        For more Lasso Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default ‘neg_mean_squared_error’

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "lasso"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "lasso_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False        
        
        alpha : float, optional
            Constant that multiplies the L1 term.
            Defaults to 1.0. alpha = 0 is equivalent to an ordinary least square, solved by the LinearRegression object.
            For numerical reasons, using alpha = 0 with the Lasso object is not advised.
            Given this, you should use the LinearRegression object.

        fit_intercept : boolean, optional, default True
            Whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).

        normalize : boolean, optional, default False
            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
            
        precompute : True | False | array-like, default=False
            Whether to use a precomputed Gram matrix to speed up calculations.
            If set to 'auto' let us decide. The Gram matrix can also be passed as argument.
            For sparse input this option is always True to preserve sparsity.

        max_iter : int, optional
            The maximum number of iterations
        
        tol : float, optional
            The tolerance for the optimization:
             if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol.
        
        positive : bool, optional
            When set to True, forces the coefficients to be positive.

        selection : str, default ‘cyclic’
            If set to ‘random’, a random coefficient is updated every iteration rather than looping over features sequentially by default.
            This (setting to ‘random’) often leads to significantly faster convergence especially when tol is higher than 1e-4.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["lasso"]

        model = Lasso(**kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def ridge_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="ridge_reg",
        new_col_name="ridge_reg_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Ridge Regression model. 

        For more Ridge Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default ‘neg_mean_squared_error’

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "ridge"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "ridge_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False        
        
        alpha : {float, array-like}, shape (n_targets)
            Regularization strength; must be a positive float.
            Regularization improves the conditioning of the problem and reduces the variance of the estimates.
            Larger values specify stronger regularization.
            Alpha corresponds to C^-1 in other linear models such as LogisticRegression or LinearSVC.
            If an array is passed, penalties are assumed to be specific to the targets. Hence they must correspond in number.
        
        fit_intercept : boolean
            Whether to calculate the intercept for this model.
            If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered).

        normalize : boolean, optional, default False

            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
        
        max_iter : int, optional
            Maximum number of iterations for conjugate gradient solver.

        tol : float
            Precision of the solution.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["ridge_reg"]

        model = Ridge(**kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def sgd_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="sgd_reg",
        new_col_name="sgd_reg_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a SGD Regression model. 

        Linear model fitted by minimizing a regularized empirical loss with SGD

        SGD stands for Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate).

        The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net).
        If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection.

        For more SGD Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default ‘neg_mean_squared_error’

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "sgd_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "sgd_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False        
        
        loss : str, default: ‘squared_loss’
            The loss function to be used.
            
            The possible values are ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’

            The ‘squared_loss’ refers to the ordinary least squares fit.
            ‘huber’ modifies ‘squared_loss’ to focus less on getting outliers correct by switching from squared to linear loss past a distance of epsilon.
            ‘epsilon_insensitive’ ignores errors less than epsilon and is linear past that; this is the loss function used in SVR.
            ‘squared_epsilon_insensitive’ is the same but becomes squared loss past a tolerance of epsilon.

        penalty : str, ‘none’, ‘l2’, ‘l1’, or ‘elasticnet’
            The penalty (aka regularization term) to be used.
            Defaults to ‘l2’ which is the standard regularizer for linear SVM models.
            ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.

        alpha : float
            Constant that multiplies the regularization term.
            Defaults to 0.0001 Also used to compute learning_rate when set to ‘optimal’.

        l1_ratio : float
            The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
            l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
            Defaults to 0.15.

        fit_intercept : bool
            Whether the intercept should be estimated or not.
            If False, the data is assumed to be already centered.
            Defaults to True.

        max_iter : int, optional (default=1000)
            The maximum number of passes over the training data (aka epochs).
            It only impacts the behavior in the fit method, and not the partial_fit.

        tol : float or None, optional (default=1e-3)
            The stopping criterion. 
            If it is not None, the iterations will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs.

        shuffle : bool, optional
            Whether or not the training data should be shuffled after each epoch. Defaults to True.

        epsilon : float
            Epsilon in the epsilon-insensitive loss functions; only if loss is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
            
            For ‘huber’, determines the threshold at which it becomes less important to get the prediction exactly right.
            For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold.
        
        learning_rate : string, optional
            The learning rate schedule:

                ‘constant’:
                    eta = eta0

                ‘optimal’:
                    eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.

                ‘invscaling’: [default]
                    eta = eta0 / pow(t, power_t)

                ‘adaptive’:
                    eta = eta0, as long as the training keeps decreasing.
                    Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True,
                    the current learning rate is divided by 5.

        eta0 : double
            The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules.
            The default value is 0.01.

        power_t : double
            The exponent for inverse scaling learning rate [default 0.5].

        early_stopping : bool, default=False
            Whether to use early stopping to terminate training when validation score is not improving.
            If set to True, it will automatically set aside a fraction of training data as validation 
            and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.

        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set for early stopping.
            Must be between 0 and 1. Only used if early_stopping is True.

        n_iter_no_change : int, default=5
            Number of iterations with no improvement to wait before early stopping.

        average : bool or int, optional
            When set to True, computes the averaged SGD weights and stores the result in the coef_ attribute.
            If set to an int greater than 1, averaging will begin once the total number of samples seen reaches average.
            So average=10 will begin averaging after seeing 10 samples.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["sgd_reg"]

        model = SGDRegressor(**kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def adaboost_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="ada_reg",
        new_col_name="ada_reg_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains an AdaBoost Regression model.

        An AdaBoost classifier is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset
        but where the weights of incorrectly classified instances are adjusted such that subsequent regressors focus more on difficult cases.

        For more AdaBoost info, you can view it here:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default ‘neg_mean_squared_error’

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "ada_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "ada_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        base_estimator : object, optional (default=None)
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes.
            If None, then the base estimator is DecisionTreeRegressor(max_depth=3)

        n_estimators : integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.

        learning_rate : float, optional (default=1.)
            Learning rate shrinks the contribution of each classifier by learning_rate.
            There is a trade-off between learning_rate and n_estimators.

        loss : {‘linear’, ‘square’, ‘exponential’}, optional (default=’linear’)
            The loss function to use when updating the weights after each boosting iteration.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["ada_reg"]

        model = AdaBoostRegressor(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def bagging_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="bag_reg",
        new_col_name="bag_reg_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Bagging Regressor model.

        A Bagging classifier is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
        Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

        For more Bagging Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default ‘neg_mean_squared_error’

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "bag_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "bag_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        base_estimator : object or None, optional (default=None)
            The base estimator to fit on random subsets of the dataset.
            If None, then the base estimator is a decision tree.

        n_estimators : int, optional (default=10)
            The number of base estimators in the ensemble.

        max_samples : int or float, optional (default=1.0)
            The number of samples to draw from X to train each base estimator.

                If int, then draw max_samples samples.
                If float, then draw max_samples * X.shape[0] samples.

        max_features : int or float, optional (default=1.0)
            The number of features to draw from X to train each base estimator.

                If int, then draw max_features features.
                If float, then draw max_features * X.shape[1] features.

        bootstrap : boolean, optional (default=True)
            Whether samples are drawn with replacement. If False, sampling without replacement is performed.

        bootstrap_features : boolean, optional (default=False)
            Whether features are drawn with replacement.

        oob_score : bool, optional (default=False)
            Whether to use out-of-bag samples to estimate the generalization error.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["bag_reg"]

        model = BaggingRegressor(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def gradient_boosting_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="grad_reg",
        new_col_name="grad_reg_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Gradient Boosting regression model.

        GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.
        In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. 

        For more Gradient Boosting Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'neg_mean_squared_error'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "grad_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "grad_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        loss : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}, optional (default=’ls’)
            loss function to be optimized.
            
            ‘ls’ refers to least squares regression.
            ‘lad’ (least absolute deviation) is a highly robust loss function solely based on order information of the input variables.
            ‘huber’ is a combination of the two.
            ‘quantile’ allows quantile regression (use alpha to specify the quantile).
            
        learning_rate : float, optional (default=0.1)
            learning rate shrinks the contribution of each tree by learning_rate.
            There is a trade-off between learning_rate and n_estimators.

        n_estimators : int (default=100)
            The number of boosting stages to perform.
            Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.

        subsample : float, optional (default=1.0)
            The fraction of samples to be used for fitting the individual base learners.
            If smaller than 1.0 this results in Stochastic Gradient Boosting.
            Subsample interacts with the parameter n_estimators.
            Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.

        criterion : string, optional (default=”friedman_mse”)
            The function to measure the quality of a split.
            Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error.
            The default value of “friedman_mse” is generally the best as it can provide a better approximation in some cases.

        min_samples_split : int, float, optional (default=2)
            The minimum number of samples required to split an internal node:

                If int, then consider min_samples_split as the minimum number.
                If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

        min_samples_leaf : int, float, optional (default=1)
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
            This may have the effect of smoothing the model, especially in regression.

                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

        max_depth : integer, optional (default=3)
            maximum depth of the individual regression estimators.
            The maximum depth limits the number of nodes in the tree.
            Tune this parameter for best performance; the best value depends on the interaction of the input variables.

        max_features : int, float, string or None, optional (default=None)
            The number of features to consider when looking for the best split:

                If int, then consider max_features features at each split.
                If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
                If “auto”, then max_features=sqrt(n_features).
                If “sqrt”, then max_features=sqrt(n_features).
                If “log2”, then max_features=log2(n_features).
                If None, then max_features=n_features.

            Choosing max_features < n_features leads to a reduction of variance and an increase in bias.

            Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features. 

        alpha : float (default=0.9)
            The alpha-quantile of the huber loss function and the quantile loss function.
            Only if loss='huber' or loss='quantile'.

        max_leaf_nodes : int or None, optional (default=None)
            Grow trees with max_leaf_nodes in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        presort : bool or ‘auto’, optional (default=’auto’)
            Whether to presort the data to speed up the finding of best splits in fitting.
            Auto mode by default will use presorting on dense data and default to normal sorting on sparse data.
            Setting presort to true on sparse data will raise an error.

        validation_fraction : float, optional, default 0.1
            The proportion of training data to set aside as validation set for early stopping.
            Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.

        tol : float, optional, default 1e-4
            Tolerance for the early stopping.
            When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["grad_reg"]

        model = GradientBoostingRegressor(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def random_forest_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="rf_reg",
        new_col_name="rf_reg_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Random Forest Regression model.

        A random forest is a meta estimator that fits a number of decision tree regressors on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
        The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).

        For more Random Forest info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 
        
        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'neg_mean_squared_error'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "rf_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "rf_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False
        
        n_estimators : integer, optional (default=10)
            The number of trees in the forest.

        criterion : string, optional (default=”mse”)
            The function to measure the quality of a split.           
            Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion, and “mae” for the mean absolute error.

        max_depth : integer or None, optional (default=None)
            The maximum depth of the tree.
            If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

        min_samples_split : int, float, optional (default=2)
            The minimum number of samples required to split an internal node:

                If int, then consider min_samples_split as the minimum number.
                If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

        min_samples_leaf : int, float, optional (default=1)
            The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.

                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

        max_features : int, float, string or None, optional (default=”auto”)
            The number of features to consider when looking for the best split:

                If int, then consider max_features features at each split.
                If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
                If “auto”, then max_features=sqrt(n_features).
                If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
                If “log2”, then max_features=log2(n_features).
                If None, then max_features=n_features.

            Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
        
        max_leaf_nodes : int or None, optional (default=None)
            Grow trees with max_leaf_nodes in best-first fashion.
            Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, optional (default=0.)
            A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

            The weighted impurity decrease equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

            where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.

            N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.

        bootstrap : boolean, optional (default=True)
            Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.

        oob_score : bool (default=False)
            Whether to use out-of-bag samples to estimate the generalization accuracy.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["rf_reg"]

        model = RandomForestRegressor(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def decision_tree_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="dt_reg",
        new_col_name="dt_reg_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Decision Tree Regression model.

        For more Decision Tree info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'neg_mean_squared_error'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "dt_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "dt_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False       	

        criterion : string, optional (default=”mse”)
            The function to measure the quality of a split.
            
            Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node,
             “friedman_mse”, which uses mean squared error with Friedman’s improvement score for potential splits,
             and “mae” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node.

        splitter : string, optional (default=”best”)
            The strategy used to choose the split at each node.
            Supported strategies are “best” to choose the best split and “random” to choose the best random split.

        max_depth : int or None, optional (default=None)
            The maximum depth of the tree.
            If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

        min_samples_split : int, float, optional (default=2)
            The minimum number of samples required to split an internal node:

                If int, then consider min_samples_split as the minimum number.
                If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

        min_samples_leaf : int, float, optional (default=1)
            The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.

                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

        max_features : int, float, string or None, optional (default=None)
            The number of features to consider when looking for the best split:

                    If int, then consider max_features features at each split.
                    If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
                    If “auto”, then max_features=sqrt(n_features).
                    If “sqrt”, then max_features=sqrt(n_features).
                    If “log2”, then max_features=log2(n_features).
                    If None, then max_features=n_features.

            Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.

        max_leaf_nodes : int or None, optional (default=None)
            Grow a tree with max_leaf_nodes in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, optional (default=0.)
            A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

            The weighted impurity decrease equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

            where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.

            N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.

        min_impurity_split : float, (default=1e-7)
            Threshold for early stopping in tree growth.
            A node will split if its impurity is above the threshold, otherwise it is a leaf.

        presort : bool, optional (default=False)
            Whether to presort the data to speed up the finding of best splits in fitting.
            For the default settings of a decision tree on large datasets, setting this to true may slow down the training process.
            When using either a smaller dataset or a restricted depth, this may speed up the training.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["dt_reg"]

        model = DecisionTreeRegressor(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def linearsvr(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="linsvr",
        new_col_name="linsvr_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains a Linear Support Vector Regression model.

        Similar to SVR with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm,
        so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.

        For more Support Vector info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 
        
        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'neg_mean_squared_error’

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "linsvr_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "linsvr_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False    	

        epsilon : float, optional (default=0.0)
            Epsilon parameter in the epsilon-insensitive loss function.
            Note that the value of this parameter depends on the scale of the target variable y.
            If unsure, set epsilon=0.

        tol : float, optional (default=1e-4)
            Tolerance for stopping criteria.

        C : float, optional (default=1.0)
            Penalty parameter C of the error term.

        loss : string, ‘hinge’ or ‘squared_hinge’ (default=’squared_hinge’)
            Specifies the loss function.            
            ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the square of the hinge loss.

        dual : bool, (default=True)
            Select the algorithm to either solve the dual or primal optimization problem.
            Prefer dual=False when n_samples > n_features.

        fit_intercept : boolean, optional (default=True)
            Whether to calculate the intercept for this model.
            If set to false, no intercept will be used in calculations (i.e. data is expected to be already centered).

        intercept_scaling : float, optional (default=1)
            When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance vector.
            The intercept becomes intercept_scaling * synthetic feature weight Note! the synthetic feature weight is subject to l1/l2 regularization as all other features.
            To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
       
        max_iter : int, (default=1000)
            The maximum number of iterations to be run.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["linsvr"]

        model = LinearSVR(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    @add_to_queue
    def svr(
        self,
        cv=None,
        gridsearch=None,
        score='neg_mean_squared_error',
        learning_curve=False,
        model_name="svr",
        new_col_name="svr_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Epsilon-Support Vector Regression.

        The free parameters in the model are C and epsilon.

        The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples.
        For large datasets consider using model.linearsvr or model.sgd_regression instead

        For more Support Vector info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’

        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'neg_mean_squared_error'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "linsvr"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "linsvr_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False    	

        kernel : string, optional (default=’rbf’)
            Specifies the kernel type to be used in the algorithm.
            It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
            If none is given, ‘rbf’ will be used.
            If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).

        degree : int, optional (default=3)
            Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

        gamma : float, optional (default=’auto’)
            Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            Current default is ‘auto’ which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma.

        coef0 : float, optional (default=0.0)
            Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

        tol : float, optional (default=1e-3)
            Tolerance for stopping criterion.

        C : float, optional (default=1.0)
            Penalty parameter C of the error term.

        epsilon : float, optional (default=0.1)
            Epsilon in the epsilon-SVR model.
            It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.

        shrinking : boolean, optional (default=True)
            Whether to use the shrinking heuristic.

        cache_size : float, optional
            Specify the size of the kernel cache (in MB).

        max_iter : int, optional (default=-1)
            Hard limit on iterations within solver, or -1 for no limit.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        report_info = technique_reason_repo["model"]["regression"]["svr"]

        model = SVR(**kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            **kwargs
        )

        return model

    @add_to_queue
    def xgboost_regression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        learning_curve=False,
        model_name="xgb_reg",
        new_col_name="xgb_reg_predictions",
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Trains an XGBoost Regression Model.

        XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
        It implements machine learning algorithms under the Gradient Boosting framework.
        XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.
        The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.

        For more XGBoost info, you can view it here: https://xgboost.readthedocs.io/en/latest/ and
        https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst. 

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE 
            - ‘neg_mean_squared_error’ --> MSE
            - ‘neg_mean_squared_log_error’ --> MSLE 
            - ‘neg_median_absolute_error’ --> MeAE 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'neg_mean_squared_error'

        learning_curve : bool, optional
            When running cross validation, True to display a learning curve, by default False

        model_name : str, optional
            Name for this model, by default "xgb_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "xgb_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False    	

        max_depth : int
            Maximum tree depth for base learners. By default 3

        learning_rate : float
            Boosting learning rate (xgb's "eta"). By default 0.1

        n_estimators : int
            Number of trees to fit. By default 100.

        objective : string or callable
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            By default, reg:linear

        booster: string
            Specify which booster to use: gbtree, gblinear or dart. By default 'gbtree'

        tree_method: string
            Specify which tree method to use
            If this parameter is set to default, XGBoost will choose the most conservative option
            available.  It's recommended to study this option from parameters
            document. By default 'auto'

        gamma : float
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
            By default 0

        subsample : float
            Subsample ratio of the training instance.
            By default 1
        
        reg_alpha : float (xgb's alpha)
            L1 regularization term on weights. By default 0

        reg_lambda : float (xgb's lambda)
            L2 regularization term on weights. By default 1

        scale_pos_weight : float
            Balancing of positive and negative weights. By default 1

        base_score:
            The initial prediction score of all instances, global bias. By default 0

        missing : float, optional
            Value in the data which needs to be present as a missing value. If
            None, defaults to np.nan.
            By default, None

        num_parallel_tree: int
            Used for boosting random forest.
            By default 1

        importance_type: string, default "gain"
            The feature importance type for the feature_importances\\_ property:
            either "gain", "weight", "cover", "total_gain" or "total_cover".
            By default 'gain'.

        Note
        ----
        A custom objective function can be provided for the ``objective``
        parameter. In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``:

        y_true: array_like of shape [n_samples]
            The target values

        y_pred: array_like of shape [n_samples]
            The predicted values

        grad: array_like of shape [n_samples]
            The value of the gradient for each sample point.

        hess: array_like of shape [n_samples]
            The value of the second derivative for each sample point

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        """

        random_state = kwargs.pop("random_state", 42)
        report_info = technique_reason_repo["model"]["regression"]["xgb_reg"]

        model = xgb.XGBRegressor(random_state=random_state, **kwargs)

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            learning_curve=learning_curve,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

        return model

    ################### CLASSIFICATION MODELS ########################

    def _run_supervised_model(
        self,
        model,
        model_name,
        model_type,
        new_col_name,
        report_info,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Helper function that generalizes model orchestration.
        """

        random_state = kwargs.pop("random_state", 42)
        cv, kwargs = _get_cv_type(cv, random_state, **kwargs)

        if cv:
            cv_scores = run_crossvalidation(
                model,
                self._data_properties.x_train,
                self._y_train,
                cv=cv,
                scoring=score,
                learning_curve=learning_curve,
            )

            # NOTE: Not satisified with this implementation, which is why this whole process needs a rework but is satisfactory... for a v1.
            if not run:
                return cv_scores

        if gridsearch:
            cv = cv if cv else 5
            model = run_gridsearch(model, gridsearch, cv, score, verbose=verbose)

        # Train a model and predict on the test test.
        model.fit(self._data_properties.x_train, self._y_train)

        self._train_result_data[new_col_name] = model.predict(
            self._data_properties.x_train
        )

        if self._data_properties.x_test is not None:
            self._test_result_data[new_col_name] = model.predict(
                self._data_properties.x_test
            )

        # Report the results
        if self.report is not None:
            if gridsearch:
                self.report.report_gridsearch(model, verbose)

            self.report.report_technique(report_info)

        if gridsearch:
            model = model.best_estimator_

        self._models[model_name] = model_type(self, model_name, model, new_col_name)

        return self._models[model_name]

    # TODO: Consider whether gridsearch/cv is necessary
    def _run_unsupervised_model(
        self,
        model,
        model_name,
        new_col_name,
        report_info,
        cv=None,
        gridsearch=None,
        score="accuracy",
        learning_curve=False,
        run=False,
        verbose=2,
        **kwargs
    ):
        """
        Helper function that generalizes model orchestration.
        """

        random_state = kwargs.pop("random_state", 42)
        cv, kwargs = _get_cv_type(cv, random_state, **kwargs)

        if cv:
            cv_scores = run_crossvalidation(
                model,
                self._data_properties.x_train,
                self._y_train,
                cv=cv,
                scoring=score,
                learning_curve=learning_curve,
            )

            # NOTE: Not satisified with this implementation, which is why this whole process needs a rework but is satisfactory... for a v1.
            if not run:
                return cv_scores

        if gridsearch:
            cv = cv if cv else 5
            model = run_gridsearch(model, gridsearch, cv, score, verbose=verbose)

        self._train_result_data[new_col_name] = model.fit_predict(
            self._data_properties.x_train
        )

        if self._data_properties.x_test is not None:
            if hasattr(model, "predict"):
                self._test_result_data[new_col_name] = model.predict(
                    self._data_properties.x_test
                )
            else:
                warnings.warn(
                    "Model does not have a predict function, unable to predict on the test data set. Consider combining your datasets into 1 and set `model.x_test = None`"
                )

        if self.report is not None:
            if gridsearch:
                self.report.report_gridsearch(model, verbose)

            self.report.report_technique(report_info)

        if gridsearch:
            model = model.best_estimator_

        self._models[model_name] = UnsupervisedModel(
            self, model_name, model, new_col_name
        )

        return self._models[model_name]
