
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
    def logistic_regression(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="log_reg", new_col_name="log_predictions", run=False, verbose=2, **kwargs):
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
        """
 
        random_state = kwargs.pop('random_state', 42)
        solver = kwargs.pop('solver', 'lbfgs')
        report_info = technique_reason_repo['model']['classification']['logreg']

        model = LogisticRegression(solver=solver, random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def ridge_classification(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="ridge_cls", new_col_name="ridge_cls_predictions", run=False, verbose=2, **kwargs):
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
        """
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['ridge_cls']

        model = RidgeClassifier(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def sgd_classification(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="sgd_cls", new_col_name="sgd_cls_predictions", run=False, verbose=2, **kwargs):
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

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def adaboost_classification(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="ada_cls", new_col_name="ada_cls_predictions", run=False, verbose=2, **kwargs):
        """
        Trains an AdaBoost classification model.

        An AdaBoost [1] classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset
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
        """
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['ada_cls']

        model = AdaBoostClassifier(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def bagging_classification(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="bag_cls", new_col_name="bag_cls_predictions", run=False, verbose=2, **kwargs):
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

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def gradient_boosting_classification(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="grad_cls", new_col_name="grad_cls_predictions", run=False, verbose=2, **kwargs):
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
        """
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['grad_cls']

        model = GradientBoostingClassifier(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def isolation_forest(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="iso_forest", new_col_name="iso_predictions", run=False, verbose=2, **kwargs):
        """
        Isolation Forest Algorithm

        Return the anomaly score of each sample using the IsolationForest algorithm

        Return the anomaly score of each sample using the IsolationForest algorithm

        The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

        For more Isolation Forest info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest

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

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def random_forest_classification(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="rf_cls", new_col_name="rf_cls_predictions", run=False, verbose=2, **kwargs):
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

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['rf_cls']

        model = RandomForestClassifier(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def nb_bernoulli_classification(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="bern", new_col_name="bern_predictions", run=False, verbose=2, **kwargs):
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
        """

        report_info = technique_reason_repo['model']['classification']['bern']

        model = BernoulliNB(**kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, **kwargs)

        return model

    @add_to_queue
    def nb_gaussian_classification(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="gauss", new_col_name="gauss_predictions", run=False, verbose=2, **kwargs):
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
        """
                 
        report_info = technique_reason_repo['model']['classification']['gauss']

        model = GaussianNB(**kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, **kwargs)

        return model

    @add_to_queue
    def nb_multinomial_classification(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="multi", new_col_name="multi_predictions", run=False, verbose=2, **kwargs):
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
        """
        
        report_info = technique_reason_repo['model']['classification']['multi']

        model = MultinomialNB(**kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, **kwargs)

        return model

    @add_to_queue
    def decision_tree_classification(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="dt_cls", new_col_name="dt_cls_predictions", run=False, verbose=2, **kwargs):
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

        Returns
        -------
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['dt_cls']

        model = DecisionTreeClassifier(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    @add_to_queue
    def linearsvc(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="linsvc_cls", new_col_name="linsvc_cls_predictions", run=False, verbose=2, **kwargs):
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
            Name for this model, by default "linsvc_cls"

        new_col_name : str, optional
            Name of column for labels that are generated, by default "linsvc_cls_predictions"

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
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['linsvc_cls']

        model = LinearSVC(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model
    
    # TODO: Move this to an unsupervised model
    @add_to_queue
    def oneclass_svm(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="ocsvm", new_col_name="ocsvm_predictions", run=False, verbose=2, **kwargs):
        """
        Trains a One Class SVM model.

        Unsupervised Outlier Detection.

        For more Support Vector info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM

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
        ClassificationModel
            ClassificationModel object to view results and analyze results
        """
                 
        report_info = technique_reason_repo['model']['classification']['oneclass_cls']

        model = OneClassSVM(**kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, **kwargs)

        return model

    @add_to_queue
    def svc(self, cv=None, gridsearch=None, score='accuracy', learning_curve=False, model_name="svc", new_col_name="svc_predictions", run=False, verbose=2, **kwargs):
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
        """
                 
        random_state = kwargs.pop('random_state', 42)
        report_info = technique_reason_repo['model']['classification']['svc_cls']

        model = SVC(random_state=random_state, **kwargs)

        model = self._run_model(model, model_name, ClassificationModel, new_col_name, report_info, cv=cv, gridsearch=gridsearch, score=score, learning_curve=learning_curve, verbose=verbose, random_state=random_state, **kwargs)

        return model

    def _run_model(self, model, model_name, model_type, new_col_name, report_info, cv=None, gridsearch=None, score='accuracy', learning_curve=False, run=False, verbose=2, **kwargs):
        """
        Helper function that generalizes model orchestration.
        """

        random_state = kwargs.pop('random_state', 42)
        cv, kwargs = _get_cv_type(cv, random_state, **kwargs)
        
        if cv:
            cv_scores = run_crossvalidation(model, self._data_properties.x_train, self._y_train, cv=cv, scoring=score, learning_curve=learning_curve)

            # NOTE: Not satisified with this implementation, which is why this whole process needs a rework but is satisfactory... for a v1.
            if not run:
                return cv_scores

        if gridsearch:
            cv = cv if cv else 5            
            model = run_gridsearch(model, gridsearch, cv, score, verbose=verbose)
        
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
