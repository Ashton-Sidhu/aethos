
import multiprocessing
import warnings

from IPython import display
from pathos.multiprocessing import Pool
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              BaggingClassifier, BaggingRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso,
                                  LinearRegression, LogisticRegression, Ridge,
                                  RidgeClassifier, SGDClassifier, SGDRegressor)
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
            raise ValueError('Model {} does not exist'.format(name))

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

    # TODO: Abstract these functions out to a more general template

    @add_to_queue
    def summarize_gensim(self, *list_args, list_of_cols=[], new_col_name="_summarized", model_name="model_summarize_gensim", run=False, **summarizer_kwargs):
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
    def extract_keywords_gensim(self, *list_args, list_of_cols=[], new_col_name="_extracted_keywords", model_name="model_extracted_keywords_gensim", run=False, **keyword_kwargs):
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
    def kmeans(self, model_name="kmeans", new_col_name="kmeans_clusters", run=False, **kmeans_kwargs):
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
    def dbscan(self, model_name="dbscan", new_col_name="dbscan_clusters", run=False, **dbscan_kwargs):
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

    # NOTE: This entire process may need to be reworked.
    @add_to_queue
    def logistic_regression(self, cv=False, gridsearch=False, cv_type=5, score='accuracy', learning_curve=False, model_name="log_reg", new_col_name="log_predictions", run=False, verbose=2, **kwargs):
        """
        Trains a logistic regression model.

        For more Logistic Regression info, you can view them here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        If no Logistic Regression parameters are provided the random state is set to 42 so model results are consistent across multiple runs.

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        If using GridSearch and no grid is specified the following default grid is used:
            'penalty': ['l1', 'l2']
            'max_iter': [100, 300, 1000]
            'tol': [1e-4, 1e-3, 1e-2]
            'warm_start': [True, False]
            'C': [1e-2, 0.1, 1, 5, 10]
            'solver': ['liblinear']

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

        gridsearch : bool or dict, optional
            Parameters to gridsearch, if True, the default parameters would be used, by default False

        cv_type : int, Crossvalidation Generator, optional
            Cross validation method, by default 5

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
        """
 
        random_state = kwargs.pop('random_state', 42)
        solver = kwargs.pop('solver', 'lbfgs')
        report_info = technique_reason_repo['model']['classification']['logreg']

        model = LogisticRegression(solver=solver, random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, cv_type=cv_type, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def ridge_classification(self, cv=False, gridsearch=False, cv_type=5, score='accuracy', learning_curve=False, model_name="ridge_cls", new_col_name="ridge_cls_predictions", run=False, verbose=2, **kwargs):
        """
        Trains a Ridge Classification model.

        For more Ridge Regression parameters, you can view them here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier

        If no parameters are provided the random state is set to 42 so model results are consistent across multiple runs.

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        If using GridSearch and no grid is specified the following default grid is used:
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 3, 10]
            'max_iter': [100, 300, 1000]
            'tol': [1e-4, 1e-3, 1e-2]
            'normalize': [True, False]

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

        gridsearch : bool or dict, optional
            Parameters to gridsearch, if True, the default parameters would be used, by default False

        cv_type : int, Crossvalidation Generator, optional
            Cross validation method, by default 5

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
        """
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['ridge_cls']

        model = RidgeClassifier(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, cv_type=cv_type, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def sgd_classification(self, cv=False, gridsearch=False, cv_type=5, score='accuracy', learning_curve=False, model_name="sgd_cls", new_col_name="sgd_cls_predictions", run=False, verbose=2, **kwargs):
        """
        Trains a Linear classifier (SVM, logistic regression, a.o.) with SGD training.

        For more info please view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier

        If no parameters are provided the random state is set to 42 so model results are consistent across multiple runs.

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        If using GridSearch and no grid is specified the following default grid is used:
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 3, 10]
            'max_iter': [100, 300, 1000]
            'tol': [1e-4, 1e-3, 1e-2]
            'warm_start': 'warm_start': [True, False],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'learning_rate': ['invscaling', 'adaptive'],
            'eta0': [1e-4, 1e-3, 1e-2, 0.1],
            'epsilon': [1e-3, 1e-2, 0.1, 0]

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

        gridsearch : bool or dict, optional
            Parameters to gridsearch, if True, the default parameters would be used, by default False

        cv_type : int, Crossvalidation Generator, optional
            Cross validation method, by default 5

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

        warm_start : bool, optional
            When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.

            Repeatedly calling fit or partial_fit when warm_start is True can result in a different solution than when calling fit a single time because of the way the data is shuffled.
            If a dynamic learning rate is used, the learning rate is adapted depending on the number of samples already seen. 
            Calling fit resets this counter, while partial_fit will result in increasing the existing counter.

        average : bool or int, optional
            When set to True, computes the averaged SGD weights and stores the result in the coef_ attribute.
            If set to an int greater than 1, averaging will begin once the total number of samples seen reaches average. So average=10 will begin averaging after seeing 10 samples.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['sgd_cls']

        model = SGDClassifier(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, cv_type=cv_type, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def adaboost_classification(self, cv=False, gridsearch=False, cv_type=5, score='accuracy', learning_curve=False, model_name="ada_cls", new_col_name="ada_cls_predictions", run=False, verbose=2, **kwargs):
        """
        Trains an AdaBoost classification model.

        For more AdaBoost info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier

        If no parameters are provided the random state is set to 42 so model results are consistent across multiple runs.

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        If using GridSearch and no grid is specified the following default grid is used:
            'n_estimators': [2, 3, 5, 10, 25, 50, 100],
            'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 3, 10]

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

        gridsearch : bool or dict, optional
            Parameters to gridsearch, if True, the default parameters would be used, by default False

        cv_type : int, Crossvalidation Generator, optional
            Cross validation method, by default 5

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
        """
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['ada_cls']

        model = AdaBoostClassifier(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, cv_type=cv_type, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def bagging_classification(self, cv=False, gridsearch=False, cv_type=5, score='accuracy', learning_curve=False, model_name="bag_cls", new_col_name="bag_cls_predictions", run=False, verbose=2, **kwargs):
        """
        Trains a Bagging classification model.

        For more Bagging Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier

        If no parameters are provided the random state is set to 42 so model results are consistent across multiple runs.

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        If using GridSearch and no grid is specified the following default grid is used:
            'n_estimators': [2, 3, 5, 10, 25, 50, 100],
            'max_samples': [0.1, 0.3, 0.5, 0.7, 1],
            'max_features': [0.1, 0.3, 0.5, 0.7, 1],
            'bootstrap': [True, False],
            'bootstrap_features': [True, False],
            'oob_score': [True, False],
            'warm_start': [True, False]

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

        gridsearch : bool or dict, optional
            Parameters to gridsearch, if True, the default parameters would be used, by default False

        cv_type : int, Crossvalidation Generator, optional
            Cross validation method, by default 5

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

        warm_start : bool, optional (default=False)
            When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, 
            otherwise, just fit a whole new ensemble.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['bag_cls']

        model = BaggingClassifier(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, cv_type=cv_type, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def gradient_boosting_classification(self, cv=False, gridsearch=False, cv_type=5, score='accuracy', learning_curve=False, model_name="grad_cls", new_col_name="grad_cls_predictions", run=False, verbose=2, **kwargs):
        """
        Trains a Gradient Boosting classification model.

        For more Gradient Boosting Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

        If no parameters are provided the random state is set to 42 so model results are consistent across multiple runs.

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        If using GridSearch and no grid is specified the following default grid is used:
            'n_estimators': [2, 3, 5, 10, 25, 50, 100],
            'max_samples': [0.1, 0.3, 0.5, 0.7, 1],
            'max_features': [0.1, 0.3, 0.5, 0.7, 1],
            'bootstrap': [True, False],
            'bootstrap_features': [True, False],
            'oob_score': [True, False],
            'warm_start': [True, False]

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

        gridsearch : bool or dict, optional
            Parameters to gridsearch, if True, the default parameters would be used, by default False

        cv_type : int, Crossvalidation Generator, optional
            Cross validation method, by default 5

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
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['grad_cls']

        model = GradientBoostingClassifier(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, cv_type=cv_type, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def isolation_forest(self, cv=False, gridsearch=False, cv_type=5, score='accuracy', learning_curve=False, model_name="iso_forest", new_col_name="iso_predictions", run=False, verbose=2, **kwargs):
        """
        Isolation Forest Algorithm

        Return the anomaly score of each sample using the IsolationForest algorithm

        For more Isolation Forest info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest

        If no parameters are provided the random state is set to 42 so model results are consistent across multiple runs.

        If running cross-validation, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

        For more information regarding the cross validation methods, you can view them here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

        If using GridSearch and no grid is specified the following default grid is used:
            'n_estimators': [2, 3, 5, 10, 25, 50, 100],
            'max_features': [0.1, 0.3, 0.5, 0.7, 1],
            'max_samples': [0.1, 0.3, 0.5, 0.7, 1],
            'bootstrap': [True, False],
            'contamination: [0, 1e-5, 1e-3, 0.1, 0.5],
            'warm_start': [True, False]

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

        gridsearch : bool or dict, optional
            Parameters to gridsearch, if True, the default parameters would be used, by default False

        cv_type : int, Crossvalidation Generator, optional
            Cross validation method, by default 5

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

        warm_start : bool, optional (default=False)
            When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, 
            otherwise, just fit a whole new ensemble.

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['iso_forest']

        model = IsolationForest(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, cv_type=cv_type, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model


    def _run_model(self, model, model_name, model_type, new_col_name, report_info, cv=False, gridsearch=False, cv_type=5, score='accuracy', learning_curve=False, run=False, verbose=2, **kwargs):
        """
        Helper function that generalizes model orchestration.
        """

        random_state = kwargs.pop('random_state', 42)
        cv_type, kwargs = _get_cv_type(cv_type, random_state, **kwargs)
        
        if cv:
            cv_scores = run_crossvalidation(model, self._data_properties.x_train, self._y_train, cv=cv_type, scoring=score, learning_curve=learning_curve)

            # NOTE: Not satisified with this implementation, which is why this whole process needs a rework but is satisfactory... for a v1.
            if not run:
                return cv_scores

        if gridsearch:
            model = run_gridsearch(model, gridsearch, cv_type, score, verbose=verbose)
        
        model.fit(self._data_properties.x_train, self._y_train)

        self._train_result_data[new_col_name] = model.predict(self._data_properties.x_train)            
        
        if self._data_properties.x_test is not None:
            self._test_result_data[new_col_name] = model.predict(self._data_properties.x_test)

        if self.report is not None:
            if gridsearch:
                self.report.report_gridsearch(model, verbose)                
        
            self.report.report_technique(report_info)

        if gridsearch:
            model = model.best_estimator_

        self._models[model_name] = model_type(self, model_name, model, new_col_name)

        return self._models[model_name]
