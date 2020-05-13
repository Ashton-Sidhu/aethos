import os
import warnings
import pandas as pd
import sklearn
import numpy as np
import copy


from pathlib import Path
from IPython.display import display
from ipywidgets import widgets
from ipywidgets.widgets.widget_layout import Layout

from aethos.config import shell
from aethos.config.config import _global_config
from aethos.model_analysis.unsupervised_model_analysis import UnsupervisedModelAnalysis
from aethos.model_analysis.text_model_analysis import TextModelAnalysis
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
from aethos.util import _input_columns, split_data, _get_attr_, _get_item_

warnings.simplefilter("ignore", FutureWarning)


class ModelBase(object):
    def __init__(
        self,
        x_train,
        target,
        x_test=None,
        test_split_percentage=0.2,
        exp_name="my-experiment",
    ):

        self._models = {}
        self._queued_models = {}
        self.exp_name = exp_name

        problem = "c" if type(self).__name__ == "Classification" else "r"

        self.x_train = x_train
        self.x_test = x_test
        self.target = target
        self.test_split_percentage = test_split_percentage
        self.target_mapping = None

        if self.x_test is None and not type(self).__name__ == "Unsupervised":
            # Generate train set and test set.
            self.x_train, self.x_test = split_data(
                self.x_train, test_split_percentage, self.target, problem
            )
            self.x_train = self.x_train.reset_index(drop=True)
            self.x_test = self.x_test.reset_index(drop=True)

    def __getitem__(self, key):

        return _get_item_(self, key)

    def __getattr__(self, key):

        # For when doing multi processing when pickle is reconstructing the object
        if key in {"__getstate__", "__setstate__"}:
            return object.__getattr__(self, key)

        if key in self._models:
            return self._models[key]

        return _get_attr_(self, key)

    def __setattr__(self, key, value):

        if key not in self.__dict__ or hasattr(self, key):
            # any normal attributes are handled normally
            dict.__setattr__(self, key, value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):

        if key in self.__dict__:
            dict.__setitem__(self.__dict__, key, value)

    def __repr__(self):

        return self.x_train.to_string()

    def _repr_html_(self):  # pragma: no cover

        if self.target:
            cols = self.features + [self.target]
        else:
            cols = self.features

        return self.x_train[cols].head()._repr_html_()

    def __deepcopy__(self, memo):

        x_test = self.x_test.copy() if self.x_test is not None else None

        new_inst = type(self)(
            x_train=self.x_train.copy(),
            target=self.target,
            x_test=x_test,
            test_split_percentage=self.test_split_percentage,
            exp_name=self.exp_name,
        )

        new_inst.target_mapping = self.target_mapping
        new_inst._models = self._models
        new_inst._queued_models = self._queued_models

        return new_inst

    @property
    def features(self):
        """Features for modelling"""

        cols = self.x_train.columns.tolist()

        if self.target:
            cols.remove(self.target)

        return cols

    @property
    def train_data(self):
        """Training data used for modelling"""

        return self.x_train[self.features]

    @train_data.setter
    def train_data(self, val):
        """Setting for train_data"""

        val[self.target] = self.y_train
        self.x_train = val

    @property
    def test_data(self):
        """Testing data used to evaluate models"""

        return self.x_test[self.features] if self.x_test is not None else None

    @test_data.setter
    def test_data(self, val):
        """Test data setter"""

        val[self.target] = self.y_test
        self.x_test = val

    @property
    def y_test(self):
        """
        Property function for the testing predictor variable
        """

        if self.x_test is not None:
            if self.target:
                return self.x_test[self.target]
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
            if self.target:
                self.x_test[self.target] = value
            else:
                self.target = "label"
                self.x_test["label"] = value
                print('Added a target (predictor) field (column) named "label".')

    @property
    def columns(self):
        """
        Property to return columns in the dataset.
        """

        return self.x_train.columns.tolist()

    def copy(self):
        """
        Returns deep copy of object.
        
        Returns
        -------
        Object
            Deep copy of object
        """

        return copy.deepcopy(self)

    def help_debug(self):
        """
        Displays a tips for helping debugging model outputs and how to deal with over and underfitting.

        Credit: Andrew Ng's and his book Machine Learning Yearning

        Examples
        --------
        >>> model.help_debug()
        """

        from aethos.model_analysis.constants import DEBUG_OVERFIT, DEBUG_UNDERFIT

        overfit_labels = [
            widgets.Checkbox(description=item, layout=Layout(width="100%"))
            for item in DEBUG_OVERFIT
        ]
        underfit_labels = [
            widgets.Checkbox(description=item, layout=Layout(width="100%"))
            for item in DEBUG_UNDERFIT
        ]

        overfit_box = widgets.VBox(overfit_labels)
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
        TextModelAnalysis
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

        self._models[model_name] = TextModelAnalysis(None, self.x_train, model_name)

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
        TextModelAnalysis
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

        self._models[model_name] = TextModelAnalysis(None, self.x_train, model_name)

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
        TextModelAnalysis
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

        self._models[model_name] = TextModelAnalysis(
            w2v_model, self.x_train, model_name
        )

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
        TextModelAnalysis
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

        self._models[model_name] = TextModelAnalysis(
            d2v_model, self.x_train, model_name
        )

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
        TextModelAnalysis
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

        self._models[model_name] = TextModelAnalysis(
            lda_model, self.x_train, model_name, corpus=corpus, id2word=id2word
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

        return nlp.model

    ################### HELPER FUNCTIONS ########################

    def _run_supervised_model(
        self,
        model,
        model_name,
        model_type,
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        run=True,
        verbose=1,
        **kwargs,
    ):
        """
        Helper function that generalizes model orchestration.
        """

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
            grid_cv = _get_cv_type(cv_type, 5, False) if cv_type is not None else 5

            model = run_gridsearch(model, gridsearch, grid_cv, score, verbose=verbose)

        #############################################################
        ###################### Train Model ##########################
        #############################################################

        # Train a model and predict on the test test.
        model.fit(self.train_data, self.y_train)

        #############################################################
        ############### Initialize Model Analysis ###################
        #############################################################

        if gridsearch:
            model = model.best_estimator_

        self._models[model_name] = model_type(
            model, self.x_train, self.x_test, self.target, model_name,
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

    def _run_unsupervised_model(
        self, model, model_name, run=True, **kwargs,
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

        _make_img_project_dir(model_name)

        #############################################################
        #################### Initialize Model #######################
        #############################################################

        if random_state:
            model = model(random_state=random_state, **kwargs)
        else:
            model = model(**kwargs)

        #############################################################
        ###################### Train Model ##########################
        #############################################################

        model.fit(self.train_data)

        #############################################################
        ############### Initialize Model Analysis ###################
        #############################################################

        self._models[model_name] = UnsupervisedModelAnalysis(
            model, self.x_train, model_name
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
