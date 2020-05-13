import pandas as pd

from .model import ModelBase
from aethos.config import shell
from aethos.model_analysis.classification_model_analysis import (
    ClassificationModelAnalysis,
)
from aethos.analysis import Analysis
from aethos.cleaning.clean import Clean
from aethos.preprocessing.preprocess import Preprocess
from aethos.feature_engineering.feature import Feature
from aethos.visualizations.visualizations import Visualizations
from aethos.stats.stats import Stats
from aethos.modelling.util import add_to_queue


class Classification(
    ModelBase, Analysis, Clean, Preprocess, Feature, Visualizations, Stats
):
    def __init__(
        self,
        x_train,
        target,
        x_test=None,
        test_split_percentage=0.2,
        exp_name="my-experiment",
    ):
        """
        Class to run analysis, transform your data and run Classification algorithms.

        Parameters
        -----------
        x_train: pd.DataFrame
            Training data or aethos data object

        target: str
            For supervised learning problems, the name of the column you're trying to predict.

        x_test: pd.DataFrame
            Test data, by default None

        test_split_percentage: float
            Percentage of data to split train data into a train and test set, by default 0.2.

        exp_name : str
            Experiment name to be tracked in MLFlow.
        """

        super().__init__(
            x_train,
            target,
            x_test=x_test,
            test_split_percentage=test_split_percentage,
            exp_name=exp_name,
        )

    # NOTE: This entire process may need to be reworked.
    @add_to_queue
    def LogisticRegression(
        self,
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="log_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a logistic regression model.

        For more Logistic Regression info, you can view them here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        If running grid search, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : {kfold, strat-kfold}, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "log_reg"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.LogisticRegression()
        >>> model.LogisticRegression(model_name='lg_1', C=0.001)
        >>> model.LogisticRegression(cv_type='kfold')
        >>> model.LogisticRegression(gridsearch={'C':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.LogisticRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import LogisticRegression

        solver = kwargs.pop("solver", "lbfgs")

        model = LogisticRegression

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            solver=solver,
            **kwargs,
        )

        return model

    @add_to_queue
    def RidgeClassification(
        self,
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="ridge_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Ridge Classification model.

        For more Ridge Regression parameters, you can view them here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier        

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : {kfold, strat-kfold}, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "ridge_cls"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.RidgeClassification()
        >>> model.RidgeClassification(model_name='rc_1, tol=0.001)
        >>> model.RidgeClassification(cv_type='kfold')
        >>> model.RidgeClassification(gridsearch={'alpha':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.RidgeClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import RidgeClassifier

        model = RidgeClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="sgd_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Linear classifier (SVM, logistic regression, a.o.) with SGD training.

        For more info please view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : {kfold, strat-kfold}, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'
        
        model_name : str, optional
            Name for this model, by default "sgd_cls"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        
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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results
        
        Examples
        --------
        >>> model.SGDClassification()
        >>> model.SGDClassification(model_name='rc_1, tol=0.001)
        >>> model.SGDClassification(cv_type='kfold')
        >>> model.SGDClassification(gridsearch={'alpha':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.SGDClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import SGDClassifier

        model = SGDClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="ada_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains an AdaBoost classification model.

        An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset
        but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

        For more AdaBoost info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : {kfold, strat-kfold}, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "ada_cls"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.AdaBoostClassification()
        >>> model.AdaBoostClassification(model_name='rc_1, learning_rate=0.001)
        >>> model.AdaBoostClassification(cv_type='kfold')
        >>> model.AdaBoostClassification(gridsearch={'n_estimators': [50, 100]}, cv_type='strat-kfold')
        >>> model.AdaBoostClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import AdaBoostClassifier

        model = AdaBoostClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="bag_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Bagging classification model.

        A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
        Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

        For more Bagging Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : {kfold, strat-kfold}, Crossvalidation Generator, optional
            Cross validation method, by default None

        gridsearch : dict, optional
            Parameters to gridsearch, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "bag_cls"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.BaggingClassification()
        >>> model.BaggingClassification(model_name='m1', n_estimators=100)
        >>> model.BaggingClassification(cv_type='kfold')
        >>> model.BaggingClassification(gridsearch={'n_estimators':[100, 200]}, cv_type='strat-kfold')
        >>> model.BaggingClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import BaggingClassifier

        model = BaggingClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="grad_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Gradient Boosting classification model.

        GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.
        In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. 
        Binary classification is a special case where only a single regression tree is induced.

        For more Gradient Boosting Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier   

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "grad_cls"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.GradientBoostingClassification()
        >>> model.GradientBoostingClassification(model_name='m1', n_estimators=100)
        >>> model.GradientBoostingClassification(cv_type='kfold')
        >>> model.GradientBoostingClassification(gridsearch={'n_estimators':[100, 200]}, cv_type='strat-kfold')
        >>> model.GradientBoostingClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="rf_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Random Forest classification model.

        A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
        The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).

        For more Random Forest info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "rf_cls"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        
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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.RandomForestClassification()
        >>> model.RandomForestClassification(model_name='m1', n_estimators=100)
        >>> model.RandomForestClassification(cv_type='kfold')
        >>> model.RandomForestClassification(gridsearch={'n_estimators':[100, 200]}, cv_type='strat-kfold')
        >>> model.RandomForestClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="bern",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Bernoulli Naive Bayes classification model.

        Like MultinomialNB, this classifier is suitable for discrete data.
        The difference is that while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features.

        For more Bernoulli Naive Bayes info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
        and https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes 

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : bool, optional
            If True run crossvalidation on the model, by default False.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "bern"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        
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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.BernoulliClassification()
        >>> model.BernoulliClassification(model_name='m1', binarize=0.5)
        >>> model.BernoulliClassification(cv_type='kfold')
        >>> model.BernoulliClassification(gridsearch={'fit_prior':[True, False]}, cv_type='strat-kfold')
        >>> model.BernoulliClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.naive_bayes import BernoulliNB

        model = BernoulliNB

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="gauss",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Gaussian Naive Bayes classification model.

        For more Gaussian Naive Bayes info, you can view it here: https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : bool, optional
            If True run crossvalidation on the model, by default False.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "gauss"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

        priors : array-like, shape (n_classes,)
            Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

        var_smoothing : float, optional (default=1e-9)
            Portion of the largest variance of all features that is added to variances for calculation stability.

        Returns
        -------
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.GaussianClassification()
        >>> model.GaussianClassification(model_name='m1', var_smooting=0.0003)
        >>> model.GaussianClassification(cv_type='kfold')
        >>> model.GaussianClassification(gridsearch={'var_smoothing':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.GaussianClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.naive_bayes import GaussianNB

        model = GaussianNB

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="multi",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Multinomial Naive Bayes classification model.

        The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts.
        However, in practice, fractional counts such as tf-idf may also work.

        For more Multinomial Naive Bayes info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
        and https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes 

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : bool, optional
            If True run crossvalidation on the model, by default False.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "multi"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

        alpha : float, optional (default=1.0)
            Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

        fit_prior : boolean, optional (default=True)
            Whether to learn class prior probabilities or not. If false, a uniform prior will be used.

        class_prior : array-like, size (n_classes,), optional (default=None)
            Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

        Returns
        -------
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.MultinomialClassification()
        >>> model.MultinomialClassification(model_name='m1', alpha=0.0003)
        >>> model.MultinomialClassification(cv_type='kfold')
        >>> model.MultinomialClassification(gridsearch={'alpha':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.MultinomialClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.naive_bayes import MultinomialNB

        model = MultinomialNB

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="dt_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Decision Tree classification model.

        For more Decision Tree info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "dt_cls"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1   
        	
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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.DecisionTreeClassification()
        >>> model.DecisionTreeClassification(model_name='m1', min_impurity_split=0.0003)
        >>> model.DecisionTreeClassification(cv_type='kfold')
        >>> model.DecisionTreeClassification(gridsearch={'min_impurity_split':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.DecisionTreeClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="linsvc",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Linear Support Vector classification model.

        Supports multi classification.

        Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
        This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme.

        For more Support Vector info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "linsvc"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1    	

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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.LinearSVC()
        >>> model.LinearSVC(model_name='m1', C=0.0003)
        >>> model.LinearSVC(cv_type='kfold')
        >>> model.LinearSVC(gridsearch={'C':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.LinearSVC(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.svm import LinearSVC

        model = LinearSVC

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="svc_cls",
        run=True,
        verbose=1,
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

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "linsvc_cls"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1    	

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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.SVC()
        >>> model.SVC(model_name='m1', C=0.0003)
        >>> model.SVC(cv_type='kfold')
        >>> model.SVC(gridsearch={'C':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.SVC(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.svm import SVC

        model = SVC

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="xgb_cls",
        run=True,
        verbose=1,
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

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "xgb_cls"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1    	

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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.XGBoostClassification()
        >>> model.XGBoostClassification(model_name='m1', reg_alpha=0.0003)
        >>> model.XGBoostClassification(cv_type='kfold')
        >>> model.XGBoostClassification(gridsearch={'reg_alpha':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.XGBoostClassification(run=False) # Add model to the queue
        """
        # endregion

        import xgboost as xgb

        objective = kwargs.pop(
            "objective",
            "binary:logistic" if len(self.y_train.unique()) == 2 else "multi:softprob",
        )

        model = xgb.XGBClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="lgbm_cls",
        run=True,
        verbose=1,
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

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        cv_type : bool, optional
            If True run crossvalidation on the model, by default None.

        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None

        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'

        model_name : str, optional
            Name for this model, by default "lgbm_cls"   

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1    	

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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.LightGBMClassification()
        >>> model.LightGBMClassification(model_name='m1', reg_alpha=0.0003)
        >>> model.LightGBMClassification(cv_type='kfold')
        >>> model.LightGBMClassification(gridsearch={'reg_alpha':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.LightGBMClassification(run=False) # Add model to the queue
        """
        # endregion

        import lightgbm as lgb

        objective = kwargs.pop(
            "objective", "binary" if len(self.y_train.unique()) == 2 else "multiclass",
        )

        model = lgb.LGBMClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            objective=objective,
            **kwargs,
        )

        return model
