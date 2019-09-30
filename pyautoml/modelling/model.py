
import multiprocessing
import warnings

from IPython import display
from pathos.multiprocessing import Pool
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LogisticRegression

from pyautoml.base import SHELL, MethodBase, technique_reason_repo
from pyautoml.modelling.default_gridsearch_params import *
from pyautoml.modelling.model_analysis import *
from pyautoml.modelling.text import *
from pyautoml.modelling.util import (_run_models_parallel, add_to_queue,
                                     run_gridsearch)
from pyautoml.util import (_contructor_data_properties, _input_columns,
                           _set_item, _validate_model_name)

# TODO: For classification implement probability predictions

class Model(MethodBase):

    def __init__(self, step=None, x_train=None, x_test=None, test_split_percentage=0.2, split=True, target_field="", report_name=None):
        
        _data_properties = _contructor_data_properties(step)

        if _data_properties is None:        
            super().__init__(x_train=x_train, x_test=x_test, test_split_percentage=test_split_percentage,
                        split=split, target_field=target_field, target_mapping=None, report_name=report_name)
        else:
            super().__init__(x_train=_data_properties.x_train, x_test=_data_properties.x_test, test_split_percentage=test_split_percentage,
                        split=_data_properties.split, target_field=_data_properties.target_field, target_mapping=_data_properties.target_mapping, report_name=_data_properties.report_name)
                        
        if self._data_properties.report is not None:
            self.report.write_header("Modelling")

        self._train_result_data = self._data_properties.x_train.copy()
        self._test_result_data = self._data_properties.x_test.copy() if self._data_properties.x_test is not None else None

        if self._data_properties.target_field:
            if split:
                if isinstance(step, Model):
                    self._y_train = step._y_train
                    self._y_test = step._y_test
                    self._data_properties.x_train = step._data_properties.x_train
                    self._data_properties.x_test = step._data_properties.x_test
                else:
                    self._y_train = self._data_properties.x_train[self._data_properties.target_field]
                    self._y_test = self._data_properties.x_test[self._data_properties.target_field]
                    self._data_properties.x_train = self._data_properties.x_train.drop([self._data_properties.target_field], axis=1)
                    self._data_properties.x_test = self._data_properties.x_test.drop([self._data_properties.target_field], axis=1)
            else:
                if isinstance(step, Model):
                    self._y_train = step._y_train
                    self._data_properties.x_train = step._data_properties.x_train
                else:
                    self._y_train = self._data_properties.x_train[self._data_properties.target_field]
                    self._data_properties.x_train = self._data_properties.x_train.drop([self._data_properties.target_field], axis=1)

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
        if key in {'__getstate__', '__setstate__'}:
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
        
        if key not in self.__dict__:       # any normal attributes are handled normally
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
                        raise ValueError("Length of list: {} does not equal the number rows as the training set or test set.".format(str(len(value))))

                    self._train_result_data, self._test_result_data = _set_item(
                        self._train_result_data, self._test_result_data, key, value, x_train_length, x_test_length)

                elif isinstance(value, tuple):
                    for data in value:
                        if len(data) != x_train_length and len(data) != x_test_length:
                            raise ValueError("Length of list: {} does not equal the number rows as the training set or test set.".format(str(len(data))))

                        self._train_result_data, self._test_result_data = _set_item(
                            self._train_result_data, self._test_result_data, key, data, x_train_length, x_test_length)

                else:
                    self._train_result_data[key] = value
                    self._test_result_data[key] = value

                return self._test_result_data.head()

    def __repr__(self):

        if SHELL == 'ZMQInteractiveShell':
            
            display(self._train_result_data.head()) # Hack for jupyter notebooks

            return ''
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
    
    def run_models(self, method='parallel'):
        """
        Runs all queued models.

        The models can either be run one after the other ('series') or at the same time in parallel.

        Parameters
        ----------
        method : str, optional
            How to run models, can either be in 'series' or in 'parallel', by default 'parallel'
        """
        
        num_ran_models = len(self._models)
        
        if method == 'parallel':
            _run_models_parallel(self)
        elif method == 'series':
            for model in self._queued_models:
                self._queued_models[model]()
        else:
            raise ValueError('Invalid run method, accepted run methods are either "parallel" or "series".')

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

        results_table = pd.concat(results, axis=1, join='inner')

        def _highlight_optimal(x):
            
            if 'loss' in x.name.lower():
                is_min = x == x.min()
                return ['background-color: green' if v else '' for v in is_min]
            else:
                is_max = x == x.max()
                return ['background-color: green' if v else '' for v in is_max]

        results_table = results_table.style.apply(_highlight_optimal, axis=1)

        return results_table

    @add_to_queue
    def summarize_gensim(self, *list_args, list_of_cols=[], new_col_name="_summarized", model_name="model_summarize_gensim", run=True, **summarizer_kwargs):
        """
        Summarize bodies of text using Gensim's Text Rank algorith. Note that it uses a Text Rank variant as stated here:
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
    
        report_info = technique_reason_repo['model']['text']['textrank_summarizer']

        list_of_cols = _input_columns(list_args, list_of_cols)

        self._train_result_data, self._test_result_data = gensim_textrank_summarizer(
                x_train=self._data_properties.x_train, x_test=self._data_properties.x_test, list_of_cols=list_of_cols, new_col_name=new_col_name, **summarizer_kwargs)

        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = TextModel(self, model_name)

        return self._models[model_name]        

    @add_to_queue
    def extract_keywords_gensim(self, *list_args, list_of_cols=[], new_col_name="_extracted_keywords", model_name="model_extracted_keywords_gensim", run=True, **keyword_kwargs):
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

        report_info = technique_reason_repo['model']['text']['textrank_keywords']
        list_of_cols = _input_columns(list_args, list_of_cols)

        self._train_result_data, self._test_result_data = gensim_textrank_keywords(
                x_train=self._data_properties.x_train, x_test=self._data_properties.x_test, list_of_cols=list_of_cols, new_col_name=new_col_name, **keyword_kwargs)

        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = TextModel(self, model_name)

        return self._models[model_name]

    @add_to_queue
    def kmeans(self, model_name="kmeans", new_col_name="kmeans_clusters", run=True, **kmeans_kwargs):
        """
        K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.

        the objective of K-means is simple: group similar data points together and discover underlying patterns.
        To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset.

        In other words, the K-means algorithm identifies k number of centroids,
        and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.

        For a list of all possible options for K Means clustering please visit: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 
        
        Parameters
        ----------
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        model_name : str, optional
            Name for this model, by default "kmeans"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "kmeans_clusters"

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
        ClusterModel
            ClusterModel object to view results and further analysis
        """

        kmeans = KMeans(**kmeans_kwargs)

        report_info = technique_reason_repo['model']['unsupervised']['kmeans']

        kmeans.fit(self._data_properties.x_train)

        self._train_result_data[new_col_name] = kmeans.labels_
        
        if self._data_properties.x_test is not None:
            self._test_result_data[new_col_name] = kmeans.predict(self._data_properties.x_test)
        
        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = ClusterModel(self, model_name, kmeans, new_col_name)

        return self._models[model_name]

    @add_to_queue
    def dbscan(self, model_name="dbscan", new_col_name="dbscan_clusters", run=True, **dbscan_kwargs):
        """
        Based on a set of points (let’s think in a bidimensional space as exemplified in the figure), 
        DBSCAN groups together points that are close to each other based on a distance measurement (usually Euclidean distance) and a minimum number of points.
        It also marks as outliers the points that are in low-density regions.

        The DBSCAN algorithm should be used to find associations and structures in data that are hard to find manually but that can be relevant and useful to find patterns and predict trends.
        
        For a list of all possible options for DBSCAN please visit: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

        Parameters
        ----------
        model_name : str, optional
            Name for this model, by default "dbscan"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "dbscan_clusters"

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
        ClusterModel
            ClusterModel object to view results and further analysis
        """

        dbscan = DBSCAN(**dbscan_kwargs)

        report_info = technique_reason_repo['model']['unsupervised']['kmeans']

        if not self._data_properties.split:
            dbscan.fit(self._data_properties.x_train)
            self._train_result_data[new_col_name] = dbscan.labels_

        else:
            warnings.warn(
                'DBSCAN has no predict method, so training and testing data was combined and DBSCAN was trained on the full data. To access results you can use `.train_result_data`.')

            full_data = self._data_properties.x_train.append(
                self._data_properties.x_test, ignore_index=True)
            dbscan.fit(full_data)
            self._train_result_data[new_col_name] = dbscan.labels_


        if self.report is not None:
            self.report.report_technique(report_info)

        self._models[model_name] = ClusterModel(self, model_name, dbscan, new_col_name)

        return self._models[model_name]

    @add_to_queue
    def logistic_regression(self, gridsearch=False, gridsearch_cv=3, gridsearch_score='accuracy', model_name="log_reg", new_col_name="log_predictions", run=True, verbose=False, **logreg_kwargs):
        """
        Trains a logistic regression model.

        If no Logistic Regression parameters are provided the random state is set to 42 so model results are consistent across multiple runs.

        If using GridSearch and no grid is specified the following default grid is used:
            'penalty': ['l1', 'l2']
            'max_iter': [100, 300, 1000]
            'tol': [1e-4, 1e-3, 1e-2]
            'warm_start': [True, False]
            'C': [1e-2, 0.1, 1, 5, 10]
            'solver': ['liblinear']

        For more Logistic Regression parameters, you can view them here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

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
            - ‘precision’ etc. 	
            - ‘recall’ etc. 	
            - ‘jaccard’ etc. 	
            - ‘roc_auc’
        
        Parameters
        ----------
        gridsearch : bool or dict, optional
            Parameters to gridsearch, if True, the default parameters would be used, by default False

        gridsearch_cv : int, optional
            Number of folds to cross validate model, by default 3

        gridsearch_score : str, optional
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

        random_state : int, RandomState instance or None, optional (default=None)
            The seed of the pseudo random number generator to use when shuffling the data.
            If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random. Used when solver == ‘sag’ or ‘liblinear’.
        
        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """

        report_info = technique_reason_repo['model']['classification']['logreg']
        random_state = logreg_kwargs.pop('random_state', 42)
        solver = logreg_kwargs.pop('solver', 'lbfgs')

        if gridsearch:
            log_reg = LogisticRegression(random_state=random_state)
            log_reg = run_gridsearch(log_reg, gridsearch, logreg_gridsearch, gridsearch_cv, gridsearch_score)
        else:
            log_reg = LogisticRegression(solver=solver, random_state=random_state, **logreg_kwargs)
        
        log_reg.fit(self._data_properties.x_train, self._y_train)

        self._train_result_data[new_col_name] = log_reg.predict(self._data_properties.x_train)            
        
        if self._data_properties.x_test is not None:
            self._test_result_data[new_col_name] = log_reg.predict(self._data_properties.x_test)

        if self.report is not None:
            if gridsearch:
                self.report.report_gridsearch(log_reg, verbose)                
        
            self.report.report_technique(report_info)

        if gridsearch:
            log_reg = log_reg.best_estimator_

        self._models[model_name] = ClassificationModel(self, model_name, log_reg, new_col_name)

        return self._models[model_name]
