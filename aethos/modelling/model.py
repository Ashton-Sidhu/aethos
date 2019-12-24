import os
import warnings
from pathlib import Path

import catboost as cb
from aethos.config import shell, technique_reason_repo
from aethos.core import Data
from aethos.modelling.constants import DEBUG_OVERFIT, DEBUG_UNDERFIT
from aethos.modelling.model_analysis import *
from aethos.modelling.text import *
from aethos.modelling.util import (
    _get_cv_type,
    _run_models_parallel,
    add_to_queue,
    run_crossvalidation,
    run_gridsearch,
    to_pickle,
)
from aethos.reporting.report import Report
from aethos.templates.template_generator import TemplateGenerator as tg
from aethos.util import _input_columns, _set_item, split_data
from aethos.visualizations.visualizations import Visualizations
from IPython.display import display
from ipywidgets import widgets
from ipywidgets.widgets.widget_layout import Layout

warnings.simplefilter("ignore", FutureWarning)


class Model(Visualizations):
    """
    Modelling class that runs models, performs cross validation and gridsearch.

    Parameters
    -----------
        x_train: aethos.Data or pd.DataFrame
            Training data or aethos data object

        x_test: pd.DataFrame
            Test data, by default None

        split: bool
            True to split your training data into a train set and a test set

        test_split_percentage: float
            Percentage of data to split train data into a train and test set.
            Only used if `split=True`

        target_field: str
            For supervised learning problems, the name of the column you're trying to predict.

        report_name: str
            Name of the report to generate, by default None
    """

    def __init__(
        self,
        x_train,
        x_test=None,
        split=True,
        test_split_percentage=0.2,
        target_field="",
        report_name=None,
    ):
        step = x_train

        if isinstance(x_train, pd.DataFrame):
            self.x_train = x_train
            self.x_test = x_test
            self.split = split
            self.target_field = target_field
            self.target_mapping = None
            self.report_name = report_name
            self.test_split_percentage = test_split_percentage
        else:
            self.x_train = step.x_train
            self.x_test = step.x_test
            self.test_split_percentage = step.test_split_percentage
            self.split = step.split
            self.target_field = step.target_field
            self.target_mapping = step.target_mapping
            self.report_name = step.report_name

        if split and x_test is None:
            # Generate train set and test set.
            self.x_train, self.x_test = split_data(self.x_train, test_split_percentage)
            self.x_train.reset_index(drop=True, inplace=True)
            self.x_test.reset_index(drop=True, inplace=True)

        if report_name is not None:
            self.report = Report(report_name)
            self.report_name = self.report.filename
        else:
            self.report = None
            self.report_name = None

        # Create a master dataset that houses training data + results
        self._train_result_data = self.x_train.copy()
        self._test_result_data = self.x_test.copy() if self.x_test is not None else None

        # For supervised learning approaches, drop the target column
        if self.target_field:
            if split:
                self.x_train = self.x_train.drop([self.target_field], axis=1)
                self.x_test = self.x_test.drop([self.target_field], axis=1)
            else:
                self.x_train = self.x_train.drop([self.target_field], axis=1)

        self._models = {}
        self._queued_models = {}

        Visualizations.__init__(self, self._train_result_data)

    def __getitem__(self, key):

        try:
            return self._train_result_data[key]

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
                return self._train_result_data[key]
            else:
                return self._test_result_data[key]

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
                self._train_result_data[key] = value

                return self._train_result_data.head()
            else:
                x_train_length = self._train_result_data.shape[0]
                x_test_length = self._test_result_data.shape[0]

                if isinstance(value, (list, np.ndarray)):
                    ## If the number of entries in the list does not match the number of rows in the training or testing
                    ## set raise a value error
                    if len(value) != x_train_length and len(value) != x_test_length:
                        raise ValueError(
                            f"Length of list: {len(value)} does not equal the number rows as the training set or test set."
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
                                f"Length of list: {len(data)} does not equal the number rows as the training set or test set."
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

        return self._train_result_data.to_string()

    def _repr_html_(self):  # pragma: no cover

        return self._train_result_data.head().to_html(
            show_dimensions=True, notebook=True
        )

    @property
    def y_train(self):
        """
        Property function for the training predictor variable
        """

        return self.x_train_results[self.target_field] if self.target_field else None

    @y_train.setter
    def y_train(self, value):
        """
        Setter function for the training predictor variable
        """

        if self.target_field:
            self.x_train_results[self.target_field] = value
        else:
            self.target_field = "label"
            self.x_train_results["label"] = value
            print('Added a target (predictor) field (column) named "label".')

    @property
    def y_test(self):
        """
        Property function for the testing predictor variable
        """

        if self.x_test_results is not None:
            if self.target_field:
                return self.x_test_results[self.target_field]
            else:
                return None
        else:
            return None

    @y_test.setter
    def y_test(self, value):
        """
        Setter function for the testing predictor variable
        """

        if self.x_test is not None:
            if self.target_field:
                self.x_test_results[self.target_field] = value
            else:
                self.target_field = "label"
                self.x_test_results["label"] = value
                print('Added a target (predictor) field (column) named "label".')

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
        tg.generate_service(project_name, f"{model_obj.model_name}.pkl")

        print("docker build -t `image_name` ./")
        print("docker run -d --name `container_name` -p `port_num`:80 `image_name`")

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

        report_info = technique_reason_repo["model"]["text"]["textrank_summarizer"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        self._train_result_data, self._test_result_data = gensim_textrank_summarizer(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
            **summarizer_kwargs,
        )

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

        report_info = technique_reason_repo["model"]["text"]["textrank_keywords"]
        list_of_cols = _input_columns(list_args, list_of_cols)

        self._train_result_data, self._test_result_data = gensim_textrank_keywords(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
            **keyword_kwargs,
        )

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

        report_info = technique_reason_repo["model"]["text"]["word2vec"]

        w2v_model = gensim_word2vec(
            x_train=self.x_train,
            x_test=self.x_test,
            prep=prep,
            col_name=col_name,
            **kwargs,
        )

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

        report_info = technique_reason_repo["model"]["text"]["doc2vec"]

        d2v_model = gensim_doc2vec(
            x_train=self.x_train,
            x_test=self.x_test,
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

        report_info = technique_reason_repo["model"]["text"]["lda"]

        (
            self._train_result_data,
            self._test_result_data,
            lda_model,
            corpus,
            id2word,
        ) = gensim_lda(
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

    ################### UNSUPERVISED MODELS ########################

    @add_to_queue
    def KMeans(
        self,
        cv=None,
        gridsearch=None,
        score="homogeneity_score",
        model_name="km",
        new_col_name="kmeans_clusters",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
        """
        NOTE: If 'n_clusters' is not provided, k will automatically be determined using an elbow plot using distortion as the mteric to find the optimal number of clusters.

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

        Examples
        --------
        >>> model.KMeans()
        >>> model.KMeans(model_name='kmean_1, n_cluster=5)
        >>> model.KMeans(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.cluster import KMeans

        def find_optk():

            from yellowbrick.cluster import KElbowVisualizer

            model = KMeans(**kwargs)

            visualizer = KElbowVisualizer(model, k=(4, 12))
            visualizer.fit(self.x_train)
            visualizer.show()

            print(f"Optimal number of clusters is {visualizer.elbow_value_}.")

            return visualizer.elbow_value_

        report_info = technique_reason_repo["model"]["unsupervised"]["kmeans"]
        n_clusters = kwargs.pop("n_clusters", None)

        if not n_clusters:
            n_clusters = find_optk()

        model = KMeans

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            n_clusters=n_clusters,
            **kwargs,
        )

        return model

    @add_to_queue
    def DBScan(
        self,
        cv=None,
        gridsearch=None,
        score="homogeneity_score",
        model_name="dbs",
        new_col_name="dbscan_clusters",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.DBScan()
        >>> model.DBScan(model_name='dbs_1, min_samples=5)
        >>> model.DBScan(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.cluster import DBSCAN

        report_info = technique_reason_repo["model"]["unsupervised"]["dbscan"]

        model = DBSCAN

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def IsolationForest(
        self,
        cv=None,
        gridsearch=None,
        score="homogeneity_score",
        model_name="iso_forest",
        new_col_name="iso_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.IsolationForest()
        >>> model.IsolationForest(model_name='iso_1, max_features=5)
        >>> model.IsolationForest(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import IsolationForest

        report_info = technique_reason_repo["model"]["unsupervised"]["iso_forest"]

        model = IsolationForest

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def OneClassSVM(
        self,
        cv=None,
        gridsearch=None,
        score="homogeneity_score",
        model_name="ocsvm",
        new_col_name="ocsvm_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.OneClassSVM()
        >>> model.OneClassSVM(model_name='ocs_1, max_iter=100)
        >>> model.OneClassSVM(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.svm import OneClassSVM

        report_info = technique_reason_repo["model"]["unsupervised"]["oneclass_cls"]

        model = OneClassSVM

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def AgglomerativeClustering(
        self,
        cv=None,
        gridsearch=None,
        score="homogeneity_score",
        model_name="agglom",
        new_col_name="agglom_clusters",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.AgglomerativeClustering()
        >>> model.AgglomerativeClustering(model_name='ag_1, n_clusters=5)
        >>> model.AgglomerativeClustering(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.cluster import AgglomerativeClustering

        report_info = technique_reason_repo["model"]["unsupervised"]["agglom"]

        model = AgglomerativeClustering

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def MeanShift(
        self,
        cv=None,
        gridsearch=None,
        score="homogeneity_score",
        model_name="mshift",
        new_col_name="mshift_clusters",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.MeanShift()
        >>> model.MeanShift(model_name='ms_1', cluster_all=False)
        >>> model.MeanShift(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.cluster import MeanShift

        report_info = technique_reason_repo["model"]["unsupervised"]["ms"]

        model = MeanShift

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def GaussianMixtureClustering(
        self,
        cv=None,
        gridsearch=None,
        score="homogeneity_score",
        model_name="gm_cluster",
        new_col_name="gm_clusters",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.GuassianMixtureClustering()
        >>> model.GuassianMixtureClustering(model_name='gm_1, max_iter=1000)
        >>> model.GuassianMixtureClustering(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.mixture import GaussianMixture

        report_info = technique_reason_repo["model"]["unsupervised"]["em_gmm"]

        model = GaussianMixture

        model = self._run_unsupervised_model(
            model,
            model_name,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    ################### CLASSIFICATION MODELS ########################

    # NOTE: This entire process may need to be reworked.
    @add_to_queue
    def LogisticRegression(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="log_reg",
        new_col_name="log_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.LogisticRegression()
        >>> model.LogisticRegression(model_name='lg_1, C=0.001)
        >>> model.LogisticRegression(cv=10)
        >>> model.LogisticRegression(gridsearch={'C':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.LogisticRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import LogisticRegression

        solver = kwargs.pop("solver", "lbfgs")
        report_info = technique_reason_repo["model"]["classification"]["logreg"]

        model = LogisticRegression

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def RidgeClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="ridge_cls",
        new_col_name="ridge_cls_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.RidgeClassification()
        >>> model.RidgeClassification(model_name='rc_1, tol=0.001)
        >>> model.RidgeClassification(cv=10)
        >>> model.RidgeClassification(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.RidgeClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import RidgeClassifier

        report_info = technique_reason_repo["model"]["classification"]["ridge_cls"]

        model = RidgeClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def SGDClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="sgd_cls",
        new_col_name="sgd_cls_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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
        
        Examples
        --------
        >>> model.SGDClassification()
        >>> model.SGDClassification(model_name='rc_1, tol=0.001)
        >>> model.SGDClassification(cv=10)
        >>> model.SGDClassification(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.SGDClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import SGDClassifier

        report_info = technique_reason_repo["model"]["classification"]["sgd_cls"]

        model = SGDClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def ADABoostClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="ada_cls",
        new_col_name="ada_cls_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.AdaBoostClassification()
        >>> model.AdaBoostClassification(model_name='rc_1, learning_rate=0.001)
        >>> model.AdaBoostClassification(cv=10)
        >>> model.AdaBoostClassification(gridsearch={'n_estimators': [50, 100]}, cv='strat-kfold')
        >>> model.AdaBoostClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import AdaBoostClassifier

        report_info = technique_reason_repo["model"]["classification"]["ada_cls"]

        model = AdaBoostClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def BaggingClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="bag_cls",
        new_col_name="bag_cls_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.BaggingClassification()
        >>> model.BaggingClassification(model_name='m1', n_estimators=100)
        >>> model.BaggingClassification(cv=10)
        >>> model.BaggingClassification(gridsearch={'n_estimators':[100, 200]}, cv='strat-kfold')
        >>> model.BaggingClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import BaggingClassifier

        report_info = technique_reason_repo["model"]["classification"]["bag_cls"]

        model = BaggingClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def GradientBoostingClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="grad_cls",
        new_col_name="grad_cls_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.GradientBoostingClassification()
        >>> model.GradientBoostingClassification(model_name='m1', n_estimators=100)
        >>> model.GradientBoostingClassification(cv=10)
        >>> model.GradientBoostingClassification(gridsearch={'n_estimators':[100, 200]}, cv='strat-kfold')
        >>> model.GradientBoostingClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import GradientBoostingClassifier

        report_info = technique_reason_repo["model"]["classification"]["grad_cls"]

        model = GradientBoostingClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def RandomForestClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="rf_cls",
        new_col_name="rf_cls_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        ccp_alphanon-negative : float, optional (default=0.0)
            Complexity parameter used for Minimal Cost-Complexity Pruning.
            The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen.
            By default, no pruning is performed.
            See Minimal Cost-Complexity Pruning for details.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results

        Examples
        --------
        >>> model.RandomForestClassification()
        >>> model.RandomForestClassification(model_name='m1', n_estimators=100)
        >>> model.RandomForestClassification(cv=10)
        >>> model.RandomForestClassification(gridsearch={'n_estimators':[100, 200]}, cv='strat-kfold')
        >>> model.RandomForestClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import RandomForestClassifier

        report_info = technique_reason_repo["model"]["classification"]["rf_cls"]

        model = RandomForestClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def BernoulliClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="bern",
        new_col_name="bern_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.BernoulliClassification()
        >>> model.BernoulliClassification(model_name='m1', binarize=0.5)
        >>> model.BernoulliClassification(cv=10)
        >>> model.BernoulliClassification(gridsearch={'fit_prior':[True, False]}, cv='strat-kfold')
        >>> model.BernoulliClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.naive_bayes import BernoulliNB

        report_info = technique_reason_repo["model"]["classification"]["bern"]

        model = BernoulliNB

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def GaussianClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="gauss",
        new_col_name="gauss_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.GaussianClassification()
        >>> model.GaussianClassification(model_name='m1', var_smooting=0.0003)
        >>> model.GaussianClassification(cv=10)
        >>> model.GaussianClassification(gridsearch={'var_smoothing':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.GaussianClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.naive_bayes import GaussianNB

        report_info = technique_reason_repo["model"]["classification"]["gauss"]

        model = GaussianNB

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def MultinomialClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="multi",
        new_col_name="multi_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.MultinomialClassification()
        >>> model.MultinomialClassification(model_name='m1', alpha=0.0003)
        >>> model.MultinomialClassification(cv=10)
        >>> model.MultinomialClassification(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.MultinomialClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.naive_bayes import MultinomialNB

        report_info = technique_reason_repo["model"]["classification"]["multi"]

        model = MultinomialNB

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def DecisionTreeClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="dt_cls",
        new_col_name="dt_cls_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        ccp_alphanon-negative : float, optional (default=0.0)
            Complexity parameter used for Minimal Cost-Complexity Pruning.
            The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen.
            By default, no pruning is performed.
            See Minimal Cost-Complexity Pruning for details.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results

        Examples
        --------
        >>> model.DecisionTreeClassification()
        >>> model.DecisionTreeClassification(model_name='m1', min_impurity_split=0.0003)
        >>> model.DecisionTreeClassification(cv=10)
        >>> model.DecisionTreeClassification(gridsearch={'min_impurity_split':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.DecisionTreeClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.tree import DecisionTreeClassifier

        report_info = technique_reason_repo["model"]["classification"]["dt_cls"]

        model = DecisionTreeClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def LinearSVC(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="linsvc",
        new_col_name="linsvc_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.LinearSVC()
        >>> model.LinearSVC(model_name='m1', C=0.0003)
        >>> model.LinearSVC(cv=10)
        >>> model.LinearSVC(gridsearch={'C':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.LinearSVC(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.svm import LinearSVC

        report_info = technique_reason_repo["model"]["classification"]["linsvc"]

        model = LinearSVC

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def SVC(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="svc_cls",
        new_col_name="svc_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.SVC()
        >>> model.SVC(model_name='m1', C=0.0003)
        >>> model.SVC(cv=10)
        >>> model.SVC(gridsearch={'C':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.SVC(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.svm import SVC

        report_info = technique_reason_repo["model"]["classification"]["svc"]

        model = SVC

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def XGBoostClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="xgb_cls",
        new_col_name="xgb_cls_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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
            By default binary:logistic for binary classification or multi:softprob for multiclass classification

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

        Examples
        --------
        >>> model.XGBoostClassification()
        >>> model.XGBoostClassification(model_name='m1', reg_alpha=0.0003)
        >>> model.XGBoostClassification(cv=10)
        >>> model.XGBoostClassification(gridsearch={'reg_alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.XGBoostClassification(run=False) # Add model to the queue
        """
        # endregion

        import xgboost as xgb

        objective = kwargs.pop(
            "objective",
            "binary:logistic" if len(self.y_train.unique()) == 2 else "multi:softprob",
        )
        report_info = technique_reason_repo["model"]["classification"]["xgb_cls"]

        model = xgb.XGBClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            objective=objective,
            **kwargs,
        )

        return model

    @add_to_queue
    def LightGBMClassification(
        self,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="lgbm_cls",
        new_col_name="lgbm_cls_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
        """
        Trains an LightGBM Classification Model.

        LightGBM is a gradient boosting framework that uses a tree based learning algorithm.

        Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise.
        It will choose the leaf with max delta loss to grow.
        When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm.

        For more LightGBM info, you can view it here: https://github.com/microsoft/LightGBM and
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier

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

        model_name : str, optional
            Name for this model, by default "lgbm_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "lgbm_cls_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False    	

        boosting_type (string, optional (default='gbdt'))
            ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.

        num_leaves (int, optional (default=31))
            Maximum tree leaves for base learners.

        max_depth (int, optional (default=-1))
            Maximum tree depth for base learners, <=0 means no limit.

        learning_rate (float, optional (default=0.1))
            Boosting learning rate. You can use callbacks parameter of fit method to shrink/adapt learning rate in training using reset_parameter callback. Note, that this will ignore the learning_rate argument in training.

        n_estimators (int, optional (default=100))
            Number of boosted trees to fit.

        subsample_for_bin (int, optional (default=200000))
            Number of samples for constructing bins.

        objective (string, callable or None, optional (default=None))
            Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below). Default: ‘regression’ for LGBMRegressor, ‘binary’ or ‘multiclass’ for LGBMClassifier, ‘lambdarank’ for LGBMRanker.

        class_weight (dict, 'balanced' or None, optional (default=None))
            Weights associated with classes in the form {class_label: weight}. Use this parameter only for multi-class classification task; for binary classification task you may use is_unbalance or scale_pos_weight parameters. Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities. You may want to consider performing probability calibration (https://scikit-learn.org/stable/modules/calibration.html) of your model. The ‘balanced’ mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)). If None, all classes are supposed to have weight one. Note, that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

        min_split_gain (float, optional (default=0.))
            Minimum loss reduction required to make a further partition on a leaf node of the tree.

        min_child_weight (float, optional (default=1e-3))
            Minimum sum of instance weight (hessian) needed in a child (leaf).

        min_child_samples (int, optional (default=20))
            Minimum number of data needed in a child (leaf).

        subsample (float, optional (default=1.))
            Subsample ratio of the training instance.

        subsample_freq (int, optional (default=0))
            Frequence of subsample, <=0 means no enable.

        colsample_bytree (float, optional (default=1.))
            Subsample ratio of columns when constructing each tree.

        reg_alpha (float, optional (default=0.))
            L1 regularization term on weights.

        reg_lambda (float, optional (default=0.))
            L2 regularization term on weights.

        random_state (int or None, optional (default=None))
            Random number seed. If None, default seeds in C++ code will be used.

        n_jobs (int, optional (default=-1))
            Number of parallel threads.

        silent (bool, optional (default=True))
            Whether to print messages while running boosting.

        importance_type (string, optional (default='split'))
            The type of feature importance to be filled into feature_importances_. If ‘split’, result contains numbers of times the feature is used in a model. If ‘gain’, result contains total gains of splits which use the feature.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results

        Examples
        --------
        >>> model.LightGBMClassification()
        >>> model.LightGBMClassification(model_name='m1', reg_alpha=0.0003)
        >>> model.LightGBMClassification(cv=10)
        >>> model.LightGBMClassification(gridsearch={'reg_alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.LightGBMClassification(run=False) # Add model to the queue
        """
        # endregion

        import lightgbm as lgb

        objective = kwargs.pop(
            "objective", "binary" if len(self.y_train.unique()) == 2 else "multiclass",
        )
        report_info = technique_reason_repo["model"]["classification"]["lgbm_cls"]

        model = lgb.LGBMClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            objective=objective,
            **kwargs,
        )

        return model

    @add_to_queue
    def CatBoostClassification(
        self,
        cat_features=None,
        cv=None,
        gridsearch=None,
        score="accuracy",
        model_name="cb_cls",
        new_col_name="cb_cls_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
        """
        Trains an CatBoost Classification Model.

        CatBoost is an algorithm for gradient boosting on decision trees. 

        For more CatBoost info, you can view it here: https://catboost.ai/docs/concepts/about.html and
        https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list

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
        cat_features: list
            A one-dimensional array of categorical columns indices, by default None (all features are considered numerical)

        cv : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "cb_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "cb_cls_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False

        Important Params
        ----------------

        - cat_features
        - one_hot_max_size
        - learning_rate        
        - n_estimators
        - max_depth
        - subsample
        - colsample_bylevel
        - colsample_bytree
        - colsample_bynode
        - l2_leaf_reg
        - random_strength

        For more parameter information (as there is a lot) please view https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results

        Examples
        --------
        >>> model.CatBoostClassification()
        >>> model.CatBoostClassification(model_name='m1', learning_rate=0.0003)
        >>> model.CatBoostClassification(cv=10)
        >>> model.CatBoostClassification(gridsearch={'learning_rate':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.CatBoostClassification(run=False) # Add model to the queue
        """
        # endregion

        objective = kwargs.pop(
            "objective", "Logloss" if len(self.y_train.unique()) == 2 else "MultiClass",
        )
        report_info = technique_reason_repo["model"]["classification"]["cb_cls"]

        model = cb.CatBoostClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            objective=objective,
            cat_features=cat_features,
            **kwargs,
        )

        return model

    ################### REGRESSION MODELS ########################

    @add_to_queue
    def LinearRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="lin_reg",
        new_col_name="linreg_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.LinearRegression()
        >>> model.LinearRegression(model_name='m1', normalize=True)
        >>> model.LinearRegression(cv=10)
        >>> model.LinearRegression(gridsearch={'normalize':[True, False]}, cv='strat-kfold')
        >>> model.LinearRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import LinearRegression

        report_info = technique_reason_repo["model"]["regression"]["linreg"]

        model = LinearRegression

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def BayesianRidgeRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="bayridge_reg",
        new_col_name="bayridge_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.BayesianRidgeRegression()
        >>> model.BayesianRidgeRegression(model_name='alpha_1', C=0.0003)
        >>> model.BayesianRidgeRegression(cv=10)
        >>> model.BayesianRidgeRegression(gridsearch={'alpha_2':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.BayesianRidgeRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import BayesianRidge

        report_info = technique_reason_repo["model"]["regression"]["bay_reg"]

        model = BayesianRidge

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def ElasticnetRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="elastic",
        new_col_name="elastic_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.ElasticNetRegression()
        >>> model.ElasticNetRegression(model_name='m1', alpha=0.0003)
        >>> model.ElasticNetRegression(cv=10)
        >>> model.ElasticNetRegression(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.ElasticNetRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import ElasticNet

        report_info = technique_reason_repo["model"]["regression"]["el_net"]

        model = ElasticNet

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def LassoRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="lasso",
        new_col_name="lasso_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.LassoRegression()
        >>> model.LassoRegression(model_name='m1', alpha=0.0003)
        >>> model.LassoRegression(cv=10)
        >>> model.LassoRegression(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.LassoRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import Lasso

        report_info = technique_reason_repo["model"]["regression"]["lasso"]

        model = Lasso

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def RidgeRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="ridge_reg",
        new_col_name="ridge_reg_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.RidgeRegression()
        >>> model.RidgeRegression(model_name='m1', alpha=0.0003)
        >>> model.RidgeRegression(cv=10)
        >>> model.RidgeRegression(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.RidgeRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import Ridge

        report_info = technique_reason_repo["model"]["regression"]["ridge_reg"]

        model = Ridge

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def SGDRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="sgd_reg",
        new_col_name="sgd_reg_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.SGDRegression()
        >>> model.SGDRegression(model_name='m1', alpha=0.0003)
        >>> model.SGDRegression(cv=10)
        >>> model.SGDRegression(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.SGDRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import SGDRegressor

        report_info = technique_reason_repo["model"]["regression"]["sgd_reg"]

        model = SGDRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def ADABoostRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="ada_reg",
        new_col_name="ada_reg_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.AdaBoostRegression()
        >>> model.AdaBoostRegression(model_name='m1', learning_rate=0.0003)
        >>> model.AdaBoostRegression(cv=10)
        >>> model.AdaBoostRegression(gridsearch={'learning_rate':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.AdaBoostRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import AdaBoostRegressor

        report_info = technique_reason_repo["model"]["regression"]["ada_reg"]

        model = AdaBoostRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def BaggingRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="bag_reg",
        new_col_name="bag_reg_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.BaggingRegression()
        >>> model.BaggingRegression(model_name='m1', n_estimators=100)
        >>> model.BaggingRegression(cv=10)
        >>> model.BaggingRegression(gridsearch={'n_estimators':[100, 200]}, cv='strat-kfold')
        >>> model.BaggingRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import BaggingRegressor

        report_info = technique_reason_repo["model"]["regression"]["bag_reg"]

        model = BaggingRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def GradientBoostingRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="grad_reg",
        new_col_name="grad_reg_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.GradientBoostingRegression()
        >>> model.GradientBoostingRegression(model_name='m1', alpha=0.0003)
        >>> model.GradientBoostingRegression(cv=10)
        >>> model.GradientBoostingRegression(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.GradientBoostingRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import GradientBoostingRegressor

        report_info = technique_reason_repo["model"]["regression"]["grad_reg"]

        model = GradientBoostingRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def RandomForestRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="rf_reg",
        new_col_name="rf_reg_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        ccp_alphanon-negative : float, optional (default=0.0)
            Complexity parameter used for Minimal Cost-Complexity Pruning.
            The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen.
            By default, no pruning is performed.
            See Minimal Cost-Complexity Pruning for details.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results
        
        Examples
        --------
        >>> model.RandomForestRegression()
        >>> model.RandomForestRegression(model_name='m1', n_estimators=100)
        >>> model.RandomForestRegression(cv=10)
        >>> model.RandomForestRegression(gridsearch={'n_estimators':[100, 200]}, cv='strat-kfold')
        >>> model.RandomForestRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import RandomForestRegressor

        report_info = technique_reason_repo["model"]["regression"]["rf_reg"]

        model = RandomForestRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def DecisionTreeRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="dt_reg",
        new_col_name="dt_reg_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        ccp_alphanon-negative : float, optional (default=0.0)
            Complexity parameter used for Minimal Cost-Complexity Pruning.
            The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen.
            By default, no pruning is performed.
            See Minimal Cost-Complexity Pruning for details.

        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results

        Examples
        --------
        >>> model.DecisionTreeRegression()
        >>> model.DecisionTreeRegression(model_name='m1', min_impurity_split=0.0003)
        >>> model.DecisionTreeRegression(cv=10)
        >>> model.DecisionTreeRegression(gridsearch={'min_impurity_split':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.DecisionTreeRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.tree import DecisionTreeRegressor

        report_info = technique_reason_repo["model"]["regression"]["dt_reg"]

        model = DecisionTreeRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def LinearSVR(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="linsvr",
        new_col_name="linsvr_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.LinearSVR()
        >>> model.LinearSVR(model_name='m1', C=0.0003)
        >>> model.LinearSVR(cv=10)
        >>> model.LinearSVR(gridsearch={'C':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.LinearSVR(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.svm import LinearSVR

        report_info = technique_reason_repo["model"]["regression"]["linsvr"]

        model = LinearSVR

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def SVR(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="svr_reg",
        new_col_name="svr_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.SVR()
        >>> model.SVR(model_name='m1', C=0.0003)
        >>> model.SVR(cv=10)
        >>> model.SVR(gridsearch={'C':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.SVR(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.svm import SVR

        report_info = technique_reason_repo["model"]["regression"]["svr"]

        model = SVR

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def XGBoostRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="xgb_reg",
        new_col_name="xgb_reg_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
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

        Examples
        --------
        >>> model.XGBoostRegression()
        >>> model.XGBoostRegression(model_name='m1', reg_alpha=0.0003)
        >>> model.XGBoostRegression(cv=10)
        >>> model.XGBoostRegression(gridsearch={'reg_alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.XGBoostRegression(run=False) # Add model to the queue
        """
        # endregion

        import xgboost as xgb

        report_info = technique_reason_repo["model"]["regression"]["xgb_reg"]

        model = xgb.XGBRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def LightGBMRegression(
        self,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="lgbm_reg",
        new_col_name="lgbm_reg_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
        """
        Trains an LightGBM Regression Model.

        LightGBM is a gradient boosting framework that uses a tree based learning algorithm.

        Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise.
        It will choose the leaf with max delta loss to grow.
        When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm.

        For more LightGBM info, you can view it here: https://github.com/microsoft/LightGBM and
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor

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

        model_name : str, optional
            Name for this model, by default "lgbm_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "lgbm_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False    	

        boosting_type (string, optional (default='gbdt'))
            ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.

        num_leaves (int, optional (default=31))
            Maximum tree leaves for base learners.

        max_depth (int, optional (default=-1))
            Maximum tree depth for base learners, <=0 means no limit.

        learning_rate (float, optional (default=0.1))
            Boosting learning rate. You can use callbacks parameter of fit method to shrink/adapt learning rate in training using reset_parameter callback. Note, that this will ignore the learning_rate argument in training.

        n_estimators (int, optional (default=100))
            Number of boosted trees to fit.

        subsample_for_bin (int, optional (default=200000))
            Number of samples for constructing bins.

        objective (string, callable or None, optional (default=None))
            Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below). Default: ‘regression’ for LGBMRegressor, ‘binary’ or ‘multiclass’ for LGBMClassifier, ‘lambdarank’ for LGBMRanker.

        class_weight (dict, 'balanced' or None, optional (default=None))
            Weights associated with classes in the form {class_label: weight}. Use this parameter only for multi-class classification task; for binary classification task you may use is_unbalance or scale_pos_weight parameters. Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities. You may want to consider performing probability calibration (https://scikit-learn.org/stable/modules/calibration.html) of your model. The ‘balanced’ mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)). If None, all classes are supposed to have weight one. Note, that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

        min_split_gain (float, optional (default=0.))
            Minimum loss reduction required to make a further partition on a leaf node of the tree.

        min_child_weight (float, optional (default=1e-3))
            Minimum sum of instance weight (hessian) needed in a child (leaf).

        min_child_samples (int, optional (default=20))
            Minimum number of data needed in a child (leaf).

        subsample (float, optional (default=1.))
            Subsample ratio of the training instance.

        subsample_freq (int, optional (default=0))
            Frequence of subsample, <=0 means no enable.

        colsample_bytree (float, optional (default=1.))
            Subsample ratio of columns when constructing each tree.

        reg_alpha (float, optional (default=0.))
            L1 regularization term on weights.

        reg_lambda (float, optional (default=0.))
            L2 regularization term on weights.

        random_state (int or None, optional (default=None))
            Random number seed. If None, default seeds in C++ code will be used.

        n_jobs (int, optional (default=-1))
            Number of parallel threads.

        silent (bool, optional (default=True))
            Whether to print messages while running boosting.

        importance_type (string, optional (default='split'))
            The type of feature importance to be filled into feature_importances_. If ‘split’, result contains numbers of times the feature is used in a model. If ‘gain’, result contains total gains of splits which use the feature.
            
        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results

        Examples
        --------
        >>> model.LightGBMRegression()
        >>> model.LightGBMRegression(model_name='m1', reg_lambda=0.0003)
        >>> model.LightGBMRegression(cv=10)
        >>> model.LightGBMRegression(gridsearch={'reg_lambda':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.LightGBMRegression(run=False) # Add model to the queue
        """
        # endregion

        import lightgbm as lgb

        report_info = technique_reason_repo["model"]["regression"]["lgbm_reg"]

        model = lgb.LGBMRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model

    @add_to_queue
    def CatBoostRegression(
        self,
        cat_features=None,
        cv=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="cb_reg",
        new_col_name="cb_predictions",
        run=True,
        verbose=2,
        **kwargs,
    ):
        # region
        """
        Trains an CatBoost Regression Model.

        CatBoost is an algorithm for gradient boosting on decision trees. 
        
        For more CatBoost info, you can view it here: https://catboost.ai/docs/concepts/about.html and
        https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list

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

        model_name : str, optional
            Name for this model, by default "cb_reg"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "cb_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : bool, optional
            True if you want to print out detailed info about the model training, by default False    	

        
        Important Params
        ----------------

        - cat_features
        - one_hot_max_size
        - learning_rate        
        - n_estimators
        - max_depth
        - subsample
        - colsample_bylevel
        - colsample_bytree
        - colsample_bynode
        - l2_leaf_reg
        - random_strength

        For more parameter information (as there is a lot) please view https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
            
        Returns
        -------
        RegressionModel
            RegressionModel object to view results and analyze results

        Examples
        --------
        >>> model.CatBoostRegression()
        >>> model.CatBoostRegression(model_name='m1', learning_rate=0.0003)
        >>> model.CatBoostRegression(cv=10)
        >>> model.CatBoostRegression(gridsearch={'learning_rate':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.CatBoostRegression(run=False) # Add model to the queue
        """
        # endregion

        report_info = technique_reason_repo["model"]["regression"]["cb_reg"]

        model = cb.CatBoostRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModel,
            new_col_name,
            report_info,
            cv=cv,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            cat_features=cat_features,
            **kwargs,
        )

        return model

    ################### HELPER FUNCTIONS ########################

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
        run=True,
        verbose=2,
        **kwargs,
    ):
        """
        Helper function that generalizes model orchestration.
        """

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

        if random_state:
            model = model(random_state=random_state, **kwargs)
        else:
            model = model(**kwargs)

        if gridsearch:
            cv = cv if cv else 5

            if isinstance(model, cb.CatBoost):
                model.grid_search(
                    gridsearch, self.x_train, self.y_train, cv=cv, plot=True
                )

            else:
                model = run_gridsearch(model, gridsearch, cv, score, verbose=verbose)

        if cv:
            if isinstance(model, cb.CatBoost):
                if isinstance(cv, int):
                    folds = None
                    fold_count = cv
                else:
                    folds = cv
                    fold_count = None

                cv_dataset = cb.Pool(
                    self.x_train, self.y_train, cat_features=cat_features,
                )

                cv_scores = cb.cv(
                    cv_dataset, kwargs, folds=folds, fold_count=fold_count, plot="True",
                )

            else:
                cv_scores = run_crossvalidation(
                    model,
                    self.x_train,
                    self.y_train,
                    cv=cv,
                    scoring=score,
                    report=self.report,
                    model_name=model_name,
                )

            # NOTE: Not satisified with this implementation, which is why this whole process needs a rework but is satisfactory... for a v1.
            if not run:
                return

        # Train a model and predict on the test test.
        if isinstance(model, cb.CatBoost):
            if not gridsearch:
                model.fit(self.x_train, self.y_train, plot=True)
        else:
            model.fit(self.x_train, self.y_train)

        self._train_result_data[new_col_name] = model.predict(self.x_train)

        if self.x_test is not None:
            self._test_result_data[new_col_name] = model.predict(self.x_test)

        # Report the results
        if self.report is not None:
            if gridsearch:
                self.report.report_gridsearch(model, verbose)

            self.report.report_technique(report_info)

        if gridsearch and not isinstance(model, cb.CatBoost):
            model = model.best_estimator_

        if isinstance(model, cb.CatBoost):
            data = self.x_train if self.x_test is None else self.x_test
            label = self.y_train if self.x_test is None else self.y_test
            pool = cb.Pool(
                data,
                label=label,
                cat_features=cat_features,
                feature_names=list(self.x_train.columns),
            )

            shap_values = model.get_feature_importance(pool, type="ShapValues")

        self._models[model_name] = model_type(
            self, model_name, model, new_col_name, shap_values=shap_values, pool=pool,
        )

        print(model)

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
        run=True,
        verbose=2,
        **kwargs,
    ):
        """
        Helper function that generalizes model orchestration.
        """

        # Hard coding OneClassSVM due to its parent having random_state and the child not allowing it.
        random_state = kwargs.pop("random_state", None)
        if (
            not random_state
            and "random_state" in dir(model())
            and not isinstance(model(), sklearn.svm.OneClassSVM)
        ):
            random_state = 42

        cv, kwargs = _get_cv_type(cv, random_state, **kwargs)

        if random_state:
            model = model(random_state=random_state, **kwargs)
        else:
            model = model(**kwargs)

        if cv:
            cv_scores = run_crossvalidation(
                model,
                self.x_train,
                self.y_train,
                cv=cv,
                scoring=score,
                report=self.report,
                model_name=model_name,
            )

            # NOTE: Not satisified with this implementation, which is why this whole process needs a rework but is satisfactory... for a v1.
            if not run:
                return

        if gridsearch:
            if not self.target_field:
                raise ValueError(
                    "Target field (.target_field) must be set to evaluate best model against a scoring metric."
                )

            cv = cv if cv else 5
            model = run_gridsearch(model, gridsearch, cv, score, verbose=verbose)

            model.fit(self.x_train, self.y_train)

            self._train_result_data[new_col_name] = model.predict(self.x_train)

        else:
            if hasattr(model, "predict"):
                model.fit(self.x_train)

                self._train_result_data[new_col_name] = model.predict(self.x_train)
            else:
                self._train_result_data[new_col_name] = model.fit_predict(self.x_train)

        if self.x_test is not None:
            if hasattr(model, "predict"):
                self._test_result_data[new_col_name] = model.predict(self.x_test)
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

        print(model)

        return self._models[model_name]
