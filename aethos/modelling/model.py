import os
import warnings
from pathlib import Path
import pandas as pd
import sklearn
import numpy as np

from IPython.display import display
from ipywidgets import widgets
from ipywidgets.widgets.widget_layout import Layout

from aethos.config import shell
from aethos.config.config import _global_config
from aethos.modelling.model_analysis import (
    ClassificationModel,
    RegressionModel,
    UnsupervisedModel,
    TextModel,
)
from aethos.modelling import text
from aethos.modelling.util import (
    _get_cv_type,
    _make_img_project_dir,
    _run_models_parallel,
    add_to_queue,
    run_crossvalidation,
    run_gridsearch,
    to_pickle,
    track_model,
)
from aethos.templates.template_generator import TemplateGenerator as tg
from aethos.util import _input_columns, _set_item, split_data

warnings.simplefilter("ignore", FutureWarning)


class ModelBase(object):
    """
    Modelling class that runs models, performs cross validation and gridsearch.

    Parameters
    -----------
    x_train: aethos.Data or pd.DataFrame
        Training data or aethos data object

    x_test: pd.DataFrame
        Test data, by default None

    test_split_percentage: float
        Percentage of data to split train data into a train and test set.
        Only used if `split=True`

    target: str
        For supervised learning problems, the name of the column you're trying to predict.
    """

    # def __init__(
    #     self,
    #     x_train,
    #     x_test=None,
    #     split=True,
    #     test_split_percentage=0.2,
    #     target="",
    #     report_name=None,
    #     exp_name="my-experiment",
    # ):
    #     step = x_train

    #     if isinstance(x_train, pd.DataFrame):
    #         self.x_train = x_train
    #         self.x_test = x_test
    #         self.split = split
    #         self.target = target
    #         self.target_mapping = None
    #         self.report_name = report_name
    #         self.test_split_percentage = test_split_percentage
    #     else:
    #         self.x_train = step.x_train
    #         self.x_test = step.x_test
    #         self.test_split_percentage = step.test_split_percentage
    #         self.split = step.split
    #         self.target = step.target
    #         self.target_mapping = step.target_mapping
    #         self.report_name = step.report_name

    #     if self.split and self.x_test is None:
    #         # Generate train set and test set.
    #         self.x_train, self.x_test = split_data(self.x_train, test_split_percentage, self.target)
    #         self.x_train.reset_index(drop=True, inplace=True)
    #         self.x_test.reset_index(drop=True, inplace=True)

    #     if report_name is not None:
    #         self.report = Report(report_name)
    #         self.report_name = self.report.filename
    #     else:
    #         self.report = None
    #         self.report_name = None

    #     # Create a master dataset that houses training data + results
    #     self.x_train = self.x_train.copy()
    #     self.x_test = self.x_test.copy() if self.x_test is not None else None

    #     # For supervised learning approaches, drop the target column
    #     if self.target:
    #         if split:
    #             self.x_train = self.x_train.drop([self.target], axis=1)
    #             self.x_test = self.x_test.drop([self.target], axis=1)
    #         else:
    #             self.x_train = self.x_train.drop([self.target], axis=1)

    #     self._models = {}
    #     self._queued_models = {}
    #     self.exp_name = exp_name

    def __init__(
        self,
        x_train,
        target,
        x_test=None,
        test_split_percentage=0.2,
        exp_name="my-experiment",
    ):

        step = x_train

        if isinstance(x_train, pd.DataFrame):
            self.x_train = x_train
            self.x_test = x_test
            self.target = target
            self.target_mapping = None
            self.test_split_percentage = test_split_percentage

            if self.x_test is None:
                # Generate train set and test set.
                self.x_train, self.x_test = split_data(
                    self.x_train, test_split_percentage, self.target
                )
                self.x_train = self.x_train.reset_index(drop=True)
                self.x_test = self.x_test.reset_index(drop=True)
        else:
            self.x_train = step.x_train
            self.x_test = step.x_test
            self.test_split_percentage = step.test_split_percentage
            self.target = step.target
            self.target_mapping = step.target_mapping

        self._predicted_cols = []
        self._models = {}
        self._queued_models = {}
        self.exp_name = exp_name

    def __getitem__(self, key):

        try:
            return self.x_train[key]

        except Exception as e:
            raise AttributeError(e)

    def __getattr__(self, key):

        # For when doing multi processing when pickle is reconstructing the object
        if key in {"__getstate__", "__setstate__"}:
            return object.__getattr__(self, key)

        if key in self._models:
            return self._models[key]

        try:
            if not self.split:
                return self.x_train[key]
            else:
                return self.x_test[key]

        except Exception as e:
            raise AttributeError(e)

    def __setattr__(self, key, value):

        if key not in self.__dict__ or hasattr(self, key):
            # any normal attributes are handled normally
            dict.__setattr__(self, key, value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):

        if key in self.__dict__:
            dict.__setitem__(self.__dict__, key, value)
        else:
            if not self.split:
                self.x_train[key] = value

                return self.x_train.head()
            else:
                x_train_length = self.x_train.shape[0]
                x_test_length = self.x_test.shape[0]

                if isinstance(value, (list, np.ndarray)):
                    ## If the number of entries in the list does not match the number of rows in the training or testing
                    ## set raise a value error
                    if len(value) != x_train_length and len(value) != x_test_length:
                        raise ValueError(
                            f"Length of list: {len(value)} does not equal the number rows as the training set or test set."
                        )

                    self.x_train, self.x_test = _set_item(
                        self.x_train,
                        self.x_test,
                        key,
                        value,
                        x_train_length,
                        x_test_length,
                    )

                elif isinstance(value, tuple):
                    for data in value:
                        if len(data) != x_train_length and len(data) != x_test_length:
                            raise ValueError(
                                f"Length of list: {len(data)} does not equal the number rows as the training set or test set."
                            )

                        self.x_train, self.x_test = _set_item(
                            self.x_train,
                            self.x_test,
                            key,
                            data,
                            x_train_length,
                            x_test_length,
                        )

                else:
                    self.x_train[key] = value
                    self.x_test[key] = value

                return self.x_test.head()

    def __repr__(self):

        return self.x_train.to_string()

    def _repr_html_(self):  # pragma: no cover

        return self.x_train._repr_html_()

    @property
    def features(self):
        """Features for modelling"""

        return self.x_train.columns.tolist() - [self.target] - self.predicted_cols

    @property
    def train_data(self):
        """Training data used for modelling"""

        return self.x_train[self.features]

    @property
    def test_data(self):
        """Testing data used to evaluate models"""

        return (
            self.x_test[self.features] if self.x_test else "Test data does not exist."
        )

    @property
    def columns(self):
        """
        Property to return columns in the dataset.
        """

        return self.x_train.columns.tolist()

    def help_debug(self):
        """
        Displays a tips for helping debugging model outputs and how to deal with over and underfitting.

        Credit: Andrew Ng's and his book Machine Learning Yearning

        Examples
        --------
        >>> model.help_debug()
        """

        from aethos.modelling.constants import DEBUG_OVERFIT, DEBUG_UNDERFIT

        overfit_labels = []
        underfit_labels = []

        for item in DEBUG_OVERFIT:
            overfit_labels.append(
                widgets.Label(description=item, layout=Layout(width="100%"))
            )
        overfit_box = widgets.VBox(overfit_labels)

        for item in DEBUG_UNDERFIT:
            underfit_labels.append(
                widgets.Checkbox(description=item, layout=Layout(width="100%"))
            )
        underfit_box = widgets.VBox(underfit_labels)

        tab_list = [overfit_box, underfit_box]

        tab = widgets.Tab()
        tab.children = tab_list
        tab.set_title(0, "Overfit")
        tab.set_title(1, "Underfit")

        display(tab)

    def run_models(self, method="parallel"):
        """
        Runs all queued models.

        The models can either be run one after the other ('series') or at the same time in parallel.

        Parameters
        ----------
        method : str, optional
            How to run models, can either be in 'series' or in 'parallel', by default 'parallel'

        Examples
        --------
        >>> model.run_models()
        >>> model.run_models(method='series')
        """

        models = []

        if method == "parallel":
            models = _run_models_parallel(self)
        elif method == "series":
            for model in self._queued_models:
                models.append(self._queued_models[model]())
        else:
            raise ValueError(
                'Invalid run method, accepted run methods are either "parallel" or "series".'
            )

        return models

    def list_models(self):
        """
        Prints out all queued and ran models.

        Examples
        --------
        >>> model.list_models()
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

        Examples
        --------
        >>> model.delete_model('model1')
        """

        if name in self._queued_models:
            del self._queued_models[name]
        elif name in self._models:
            del self._models[name]
        else:
            raise ValueError(f"Model {name} does not exist")

        self.list_models()

    def compare_models(self):
        """
        Compare different models across every known metric for that model.
        
        Returns
        -------
        Dataframe
            Dataframe of every model and metrics associated for that model
        
        Examples
        --------
        >>> model.compare_models()
        """

        results = []

        for model in self._models:
            results.append(self._models[model].metrics())

        results_table = pd.concat(results, axis=1, join="inner")
        results_table = results_table.loc[:, ~results_table.columns.duplicated()]

        # Move descriptions column to end of dataframe.
        descriptions = results_table.pop("Description")
        results_table["Description"] = descriptions

        return results_table

    def to_pickle(self, name: str):
        """
        Writes model to a pickle file.
        
        Parameters
        ----------
        name : str
            Name of the model

        Examples
        --------
        >>> m = Model(df)
        >>> m.LogisticRegression()
        >>> m.to_pickle('log_reg')
        """

        model_obj = self._models[name]

        to_pickle(model_obj.model, model_obj.model_name)

    def to_service(self, model_name: str, project_name: str):
        """
        Creates an app.py, requirements.txt and Dockerfile in `~/.aethos/projects` and the necessary folder structure
        to run the model as a microservice.
        
        Parameters
        ----------
        model_name : str
            Name of the model to create a microservice of.

        project_name : str
            Name of the project that you want to create.

        Examples
        --------
        >>> m = Model(df)
        >>> m.LogisticRegression()
        >>> m.to_service('log_reg', 'your_proj_name')
        """

        model_obj = self._models[model_name]

        to_pickle(
            model_obj.model,
            model_obj.model_name,
            project=True,
            project_name=project_name,
        )
        tg.generate_service(
            project_name, f"{model_obj.model_name}.pkl", model_obj.model
        )

        print("To run:")
        print("\tdocker build -t `image_name` ./")
        print("\tdocker run -d --name `container_name` -p `port_num`:80 `image_name`")

    ################### TEXT MODELS ########################

    @add_to_queue
    def summarize_gensim(
        self,
        *list_args,
        list_of_cols=[],
        new_col_name="_summarized",
        model_name="model_summarize_gensim",
        run=True,
        **summarizer_kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.summarize_gensim('col1')
        >>> model.summarize_gensim('col1', run=False) # Add model to the queue
        """
        # endregion

        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = text.gensim_textrank_summarizer(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
            **summarizer_kwargs,
        )

        self._predicted_cols.append(col + new_col_name for col in list_of_cols)

        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = TextModel(self, None, model_name)

        return self._models[model_name]

    @add_to_queue
    def extract_keywords_gensim(
        self,
        *list_args,
        list_of_cols=[],
        new_col_name="_extracted_keywords",
        model_name="model_extracted_keywords_gensim",
        run=True,
        **keyword_kwargs,
    ):
        # region
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
        
        Examples
        --------
        >>> model.extract_keywords_gensim('col1')
        >>> model.extract_keywords_gensim('col1', run=False) # Add model to the queue
        """
        # endregion

        list_of_cols = _input_columns(list_args, list_of_cols)

        self.x_train, self.x_test = text.gensim_textrank_keywords(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
            **keyword_kwargs,
        )

        self._predicted_cols.append(col + new_col_name for col in list_of_cols)

        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = TextModel(self, None, model_name)

        return self._models[model_name]

    @add_to_queue
    def Word2Vec(self, col_name, prep=False, model_name="w2v", run=True, **kwargs):
        # region
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

        Examples
        --------
        >>> model.Word2Vec('col1', prep=True)
        >>> model.Word2Vec('col1', run=False) # Add model to the queue
        """
        # endregion

        w2v_model = text.gensim_word2vec(
            x_train=self.train_data,
            x_test=self.test_data,
            prep=prep,
            col_name=col_name,
            **kwargs,
        )

        self._predicted_cols.append(col_name)

        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = TextModel(self, w2v_model, model_name)

        return self._models[model_name]

    @add_to_queue
    def Doc2Vec(self, col_name, prep=False, model_name="d2v", run=True, **kwargs):
        # region
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
        
        Examples
        --------
        >>> model.Doc2Vec('col1', prep=True)
        >>> model.Doc2Vec('col1', run=False) # Add model to the queue
        """
        # endregion

        d2v_model = text.gensim_doc2vec(
            x_train=self.train_data,
            x_test=self.test_data,
            prep=prep,
            col_name=col_name,
            **kwargs,
        )

        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = TextModel(self, d2v_model, model_name)

        return self._models[model_name]

    @add_to_queue
    def LDA(self, col_name, prep=False, model_name="lda", run=True, **kwargs):
        # region
        """
        Extracts topics from your data using Latent Dirichlet Allocation.

        For more information on LDA, you can view it here https://radimrehurek.com/gensim/models/ldamodel.html.
        
        Parameters
        ----------
        col_name : str, optional
            Column name of text data that you want to summarize

        prep : bool, optional
            True to prep the data. Use when passing in raw text data.
            False if passing in text that is already prepped.
            By default False

        model_name : str, optional
            Name for this model, default to `lda`

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        num_topics: (int, optional)
            The number of requested latent topics to be extracted from the training corpus.

        distributed: (bool, optional)
            Whether distributed computing should be used to accelerate training.

        chunksize: (int, optional)
            Number of documents to be used in each training chunk.

        passes: (int, optional)
            Number of passes through the corpus during training.

        update_every: (int, optional)
            Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning.

        alpha: ({numpy.ndarray, str}, optional)

            Can be set to an 1D array of length equal to the number of expected topics that expresses our a-priori belief for the each topics’ probability. Alternatively default prior selecting strategies can be employed by supplying a string:

                    ’asymmetric’: Uses a fixed normalized asymmetric prior of 1.0 / topicno.

                    ’auto’: Learns an asymmetric prior from the corpus (not available if distributed==True).

        eta: ({float, np.array, str}, optional)

            A-priori belief on word probability, this can be:

                    scalar for a symmetric prior over topic/word probability,

                    vector of length num_words to denote an asymmetric user defined probability for each word,

                    matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination,

                    the string ‘auto’ to learn the asymmetric prior from the data.

        decay: (float, optional)
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten when each new document is examined. Corresponds to Kappa from Matthew D. Hoffman, David M. Blei, Francis Bach: “Online Learning for Latent Dirichlet Allocation NIPS’10”.

        offset: (float, optional)
            Hyper-parameter that controls how much we will slow down the first steps the first few iterations. Corresponds to Tau_0 from Matthew D. Hoffman, David M. Blei, Francis Bach: “Online Learning for Latent Dirichlet Allocation NIPS’10”.

        eval_every: (int, optional)
            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.

        iterations: (int, optional)
            Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.

        gamma_threshold: (float, optional)
            Minimum change in the value of the gamma parameters to continue iterating.

        minimum_probability: (float, optional)
            Topics with a probability lower than this threshold will be filtered out.

        random_state: ({np.random.RandomState, int}, optional)
            Either a randomState object or a seed to generate one. Useful for reproducibility.

        ns_conf: (dict of (str, object), optional)
            Key word parameters propagated to gensim.utils.getNS() to get a Pyro4 Nameserved. Only used if distributed is set to True.

        minimum_phi_value: (float, optional)
            if per_word_topics is True, this represents a lower bound on the term probabilities.per_word_topics (bool) – If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count).
        
        Returns
        -------
        TextModel
            Resulting model

        Examples
        --------
        >>> model.LDA('col1', prep=True)
        >>> model.LDA('col1', run=False) # Add model to the queue
        """
        # endregion

        (self.x_train, self.x_test, lda_model, corpus, id2word,) = text.gensim_lda(
            x_train=self.x_train,
            x_test=self.x_test,
            prep=prep,
            col_name=col_name,
            **kwargs,
        )

        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = TextModel(
            self, lda_model, model_name, corpus=corpus, id2word=id2word
        )

        return self._models[model_name]

    ################## PRE TRAINED MODELS #######################

    def pretrained_sentiment_analysis(
        self, col: str, model_type=None, new_col_name="sent_score", run=True
    ):
        # region
        """
        Uses Huggingface's pipeline to automatically run sentiment analysis on text.

        The default model is 'tf_distil_bert_for_sequence_classification_2'

        Possible model types are:

            - bert-base-uncased
            - bert-large-uncased
            - bert-base-cased
            - bert-large-cased
            - bert-base-multilingual-uncased
            - bert-base-multilingual-cased
            - bert-base-chinese
            - bert-base-german-cased
            - bert-large-uncased-whole-word-masking
            - bert-large-cased-whole-word-masking
            - bert-large-uncased-whole-word-masking-finetuned-squad
            - bert-large-cased-whole-word-masking-finetuned-squad
            - bert-base-cased-finetuned-mrpc
            - bert-base-german-dbmdz-cased
            - bert-base-german-dbmdz-uncased
            - bert-base-japanese
            - bert-base-japanese-whole-word-masking
            - bert-base-japanese-char
            - bert-base-japanese-char-whole-word-masking
            - bert-base-finnish-cased-v1
            - bert-base-finnish-uncased-v1
            - openai-gpt
            - gpt2
            - gpt2-medium
            - gpt2-large
            - gpt2-xl
            - transfo-xl-wt103
            - xlnet-base-cased
            - xlnet-large-cased
            - xlm-mlm-en-2048
            - xlm-mlm-ende-1024
            - xlm-mlm-enfr-1024
            - xlm-mlm-enro-1024
            - xlm-mlm-xnli15-1024
            - xlm-mlm-tlm-xnli15-1024
            - xlm-clm-enfr-1024
            - xlm-clm-ende-1024
            - xlm-mlm-17-1280
            - xlm-mlm-100-1280
            - roberta-base
            - roberta-large
            - roberta-large-mnli
            - distilroberta-base
            - roberta-base-openai-detector
            - roberta-large-openai-detector
            - distilbert-base-uncased
            - distilbert-base-uncased-distilled-squad
            - distilgpt2
            - distilbert-base-german-cased
            - distilbert-base-multilingual-cased
            - ctrl
            - camembert-base
            - albert-base-v1
            - albert-large-v1
            - albert-xlarge-v1
            - albert-xxlarge-v1
            - albert-base-v2
            - albert-large-v2
            - albert-xlarge-v2
            - albert-xxlarge-v2
            - t5-small
            - t5-base
            - t5-large
            - t5-3B
            - t5-11B
            - xlm-roberta-base
            - xlm-roberta-large

        Parameters
        ----------
        col : str
            Column of text to get sentiment analysis

        model_type : str, optional
            Type of model, by default None

        new_col_name : str, optional
            New column name for the sentiment scores, by default "sent_score"

        
        Returns
        -------
        TF or PyTorch of model

        Examples
        --------
        >>> m.pretrained_sentiment_analysis('col1')
        >>> m.pretrained_sentiment_analysis('col1', model_type='albert-base-v1')
        """
        # endregion

        try:
            from transformers import pipeline
        except ModuleNotFoundError as e:
            raise EnvironmentError(
                "Pre trained model dependencies have not been installed. Please run pip install aethos[ptmodels]"
            )

        nlp = pipeline("sentiment-analysis", model=model_type)

        self.x_train[new_col_name] = pd.Series(map(nlp, self.x_train[col].tolist()))

        if self.x_test is not None:
            self.x_test[new_col_name] = pd.Series(map(nlp, self.x_test[col].tolist()))

        self._predicted_cols.append(new_col_name)

        if self.report is not None:
            self.report.report_technique(report_info)

        return nlp.model

    def pretrained_question_answer(
        self,
        context_col: str,
        question_col: str,
        model_type=None,
        new_col_name="qa",
        run=True,
    ):
        # region
        """
        Uses Huggingface's pipeline to automatically run Q&A analysis on text.

        The default model is 'tf_distil_bert_for_question_answering_2'

        Possible model types are:

            - bert-base-uncased
            - bert-large-uncased
            - bert-base-cased
            - bert-large-cased
            - bert-base-multilingual-uncased
            - bert-base-multilingual-cased
            - bert-base-chinese
            - bert-base-german-cased
            - bert-large-uncased-whole-word-masking
            - bert-large-cased-whole-word-masking
            - bert-large-uncased-whole-word-masking-finetuned-squad
            - bert-large-cased-whole-word-masking-finetuned-squad
            - bert-base-cased-finetuned-mrpc
            - bert-base-german-dbmdz-cased
            - bert-base-german-dbmdz-uncased
            - bert-base-japanese
            - bert-base-japanese-whole-word-masking
            - bert-base-japanese-char
            - bert-base-japanese-char-whole-word-masking
            - bert-base-finnish-cased-v1
            - bert-base-finnish-uncased-v1
            - openai-gpt
            - gpt2
            - gpt2-medium
            - gpt2-large
            - gpt2-xl
            - transfo-xl-wt103
            - xlnet-base-cased
            - xlnet-large-cased
            - xlm-mlm-en-2048
            - xlm-mlm-ende-1024
            - xlm-mlm-enfr-1024
            - xlm-mlm-enro-1024
            - xlm-mlm-xnli15-1024
            - xlm-mlm-tlm-xnli15-1024
            - xlm-clm-enfr-1024
            - xlm-clm-ende-1024
            - xlm-mlm-17-1280
            - xlm-mlm-100-1280
            - roberta-base
            - roberta-large
            - roberta-large-mnli
            - distilroberta-base
            - roberta-base-openai-detector
            - roberta-large-openai-detector
            - distilbert-base-uncased
            - distilbert-base-uncased-distilled-squad
            - distilgpt2
            - distilbert-base-german-cased
            - distilbert-base-multilingual-cased
            - ctrl
            - camembert-base
            - ALBERT
            - albert-base-v1
            - albert-large-v1
            - albert-xlarge-v1
            - albert-xxlarge-v1
            - albert-base-v2
            - albert-large-v2
            - albert-xlarge-v2
            - albert-xxlarge-v2
            - t5-small
            - t5-base
            - t5-large
            - t5-3B
            - t5-11B
            - xlm-roberta-base
            - xlm-roberta-large

        Parameters
        ----------
        context_col : str
            Column name that contains the context for the question

        question_col : str
            Column name of the question

        model_type : str, optional
            Type of model, by default None

        new_col_name : str, optional
            New column name for the sentiment scores, by default "sent_score"
        
        Returns
        -------
        TF or PyTorch of model

        Examples
        --------
        >>> m.pretrained_question_answer('col1', 'col2')
        >>> m.pretrained_question_answer('col1', 'col2' model_type='albert-base-v1')
        """
        # endregion

        try:
            from transformers import pipeline
        except ModuleNotFoundError as e:
            raise EnvironmentError(
                "Pre trained model dependencies have not been installed. Please run pip install aethos[ptmodels]"
            )

        nlp = pipeline("question-answering", model=model_type)
        q_and_a = lambda c, q: nlp({"question": q, "context": c})

        self.x_train[new_col_name] = pd.Series(
            [
                q_and_a(context, question)
                for context, question in zip(
                    self.x_train[context_col], self.x_train[question_col],
                )
            ]
        )

        if self.x_test is not None:
            self.x_test[new_col_name] = pd.Series(
                [
                    q_and_a(context, question)
                    for context, question in zip(
                        self.x_test[context_col], self.x_test[question_col],
                    )
                ]
            )

        self._predicted_cols.append(new_col_name)

        if self.report is not None:
            self.report.report_technique(report_info)

        return nlp.model

    ################### HELPER FUNCTIONS ########################

    def _run_supervised_model(
        self,
        model,
        model_name,
        model_type,
        new_col_name,
        cv=None,
        gridsearch=None,
        score="accuracy",
        run=True,
        verbose=2,
        **kwargs,
    ):
        """
        Helper function that generalizes model orchestration.
        """

        import catboost as cb

        #############################################################
        ################## Initialize Variables #####################
        #############################################################

        # Hard coding SVR due to its parent having random_state and the child not allowing it.
        random_state = kwargs.pop("random_state", None)
        if (
            not random_state
            and hasattr(model(), "random_state")
            and not isinstance(model(), sklearn.svm.SVR)
        ):
            random_state = 42

        cat_features = kwargs.pop("cat_features", None)
        cv, kwargs = _get_cv_type(cv, random_state, **kwargs)
        shap_values = None
        pool = None
        run_id = None

        _make_img_project_dir(model_name)

        #############################################################
        #################### Initialize Model #######################
        #############################################################

        if random_state:
            model = model(random_state=random_state, **kwargs)
        else:
            model = model(**kwargs)

        #############################################################
        ####################### Gridsearch ##########################
        #############################################################

        if gridsearch:
            grid_cv = cv if cv else 5

            if isinstance(model, cb.CatBoost):
                model.grid_search(
                    gridsearch, self.train_data, self.y_train, cv=grid_cv, plot=True
                )

            else:
                model = run_gridsearch(model, gridsearch, cv, score, verbose=verbose)

        #############################################################
        #################### Cross Validation #######################
        #############################################################

        if cv:
            # Use native CatBoost cross validation
            if isinstance(model, cb.CatBoost):
                if isinstance(cv, int):
                    folds = None
                    fold_count = cv
                else:
                    folds = cv
                    fold_count = None

                cv_dataset = cb.Pool(
                    self.train_data, self.y_train, cat_features=cat_features,
                )

                cv_scores = cb.cv(
                    cv_dataset, kwargs, folds=folds, fold_count=fold_count, plot="True",
                )

            else:
                cv_scores = run_crossvalidation(
                    model,
                    self.train_data,
                    self.y_train,
                    cv=cv,
                    scoring=score,
                    report=self.report,
                    model_name=model_name,
                )

            # NOTE: Not satisified with this implementation, which is why this whole process needs a rework but is satisfactory... for a v1.
            if not run:
                return

        #############################################################
        ###################### Train Model ##########################
        #############################################################

        # Train a model and predict on the test test.
        if isinstance(model, cb.CatBoost):
            if not gridsearch:
                model.fit(self.train_data, self.y_train, plot=True)
        else:
            model.fit(self.train_data, self.y_train)

        self.x_train[new_col_name] = model.predict(self.train_data)

        if self.x_test is not None:
            self.x_test[new_col_name] = model.predict(self.test_data)

        self._predicted_cols.append(new_col_name)

        # ############################################################
        # ####################### Reporting ##########################
        # ############################################################

        # # Report the results
        # if self.report is not None:
        #     if gridsearch:
        #         self.report.report_gridsearch(model, verbose)

        #     self.report.report_technique(report_info)

        #############################################################
        ############### Initialize Model Analysis ###################
        #############################################################

        if gridsearch and not isinstance(model, cb.CatBoost):
            model = model.best_estimator_

        if isinstance(model, cb.CatBoost):
            data = self.train_data if self.x_test is None else self.test_data
            label = self.y_train if self.x_test is None else self.y_test
            pool = cb.Pool(
                data,
                label=label,
                cat_features=cat_features,
                feature_names=self.features,
            )

            shap_values = model.get_feature_importance(pool, type="ShapValues")

        self._models[model_name] = model_type(
            self,
            model_name,
            model,
            new_col_name,
            shap_values=shap_values,
            pool=pool,
            run_id=run_id,
        )

        #############################################################
        ######################## Tracking ###########################
        #############################################################

        if _global_config["track_experiments"]:  # pragma: no cover
            if random_state is not None:
                kwargs["random_state"] = random_state

            run_id = track_model(
                self.exp_name,
                model,
                model_name,
                kwargs,
                self.compare_models()[model_name].to_dict(),
            )
            self._models[model_name].run_id = run_id

        print(model)

        return self._models[model_name]

    # TODO: Consider whether gridsearch/cv is necessary
    def _run_unsupervised_model(
        self,
        model,
        model_name,
        new_col_name,
        cv=None,
        gridsearch=None,
        score="accuracy",
        run=True,
        verbose=2,
        **kwargs,
    ):
        """
        Helper function that generalizes model orchestration.
        """

        #############################################################
        ################## Initialize Variables #####################
        #############################################################

        # Hard coding OneClassSVM due to its parent having random_state and the child not allowing it.
        random_state = kwargs.pop("random_state", None)
        if (
            not random_state
            and "random_state" in dir(model())
            and not isinstance(model(), sklearn.svm.OneClassSVM)
        ):
            random_state = 42

        cv, kwargs = _get_cv_type(cv, random_state, **kwargs)

        _make_img_project_dir(model_name)

        #############################################################
        #################### Initialize Model #######################
        #############################################################

        if random_state:
            model = model(random_state=random_state, **kwargs)
        else:
            model = model(**kwargs)

        #############################################################
        ################ Gridsearch + Train Model ###################
        #############################################################

        if gridsearch:
            if not self.target:
                raise ValueError(
                    "Target field (.target) must be set to evaluate best model against a scoring metric."
                )

            grid_cv = cv if cv else 5
            model = run_gridsearch(model, gridsearch, grid_cv, score, verbose=verbose)

            model.fit(self.train_data, self.y_train)

            self.x_train[new_col_name] = model.predict(self.train_data)

        else:
            if hasattr(model, "predict"):
                model.fit(self.train_data)

                self.x_train[new_col_name] = model.predict(self.train_data)
            else:
                self.x_train[new_col_name] = model.fit_predict(self.train_data)

        if self.x_test is not None:
            if hasattr(model, "predict"):
                self.x_test[new_col_name] = model.predict(self.test_data)
            else:
                warnings.warn(
                    "Model does not have a predict function, unable to predict on the test data set. Consider combining your datasets into 1 and set `model.x_test = None`"
                )

        self._predicted_cols.append(new_col_name)

        #############################################################
        #################### Cross Validation #######################
        #############################################################

        if cv:
            cv_scores = run_crossvalidation(
                model,
                self.train_data,
                self.y_train,
                cv=cv,
                scoring=score,
                report=self.report,
                model_name=model_name,
            )

            # NOTE: Not satisified with this implementation.
            if not run:
                return

        # ############################################################
        # ####################### Reporting ##########################
        # ############################################################

        # if self.report is not None:
        #     if gridsearch:
        #         self.report.report_gridsearch(model, verbose)

        #     self.report.report_technique(report_info)

        #############################################################
        ############### Initialize Model Analysis ###################
        #############################################################

        if gridsearch:
            model = model.best_estimator_

        self._models[model_name] = UnsupervisedModel(
            self, model_name, model, new_col_name
        )

        #############################################################
        ######################## Tracking ###########################
        #############################################################

        if _global_config["track_experiments"]:  # pragma: no cover
            if random_state is not None:
                kwargs["random_state"] = random_state

            track_model(self.exp_name, model, model_name, kwargs)

        print(model)

        return self._models[model_name]
