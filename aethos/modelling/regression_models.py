import pandas as pd

from .model import ModelBase
from aethos.config import shell
from aethos.model_analysis.regression_model_analysis import RegressionModelAnalysis
from aethos.analysis import Analysis
from aethos.cleaning.clean import Clean
from aethos.preprocessing.preprocess import Preprocess
from aethos.feature_engineering.feature import Feature
from aethos.visualizations.visualizations import Visualizations
from aethos.stats.stats import Stats
from aethos.modelling.util import add_to_queue


class Regression(
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
        Class to run analysis, transform your data and run Regression algorithms.

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

    @add_to_queue
    def LinearRegression(
        self,
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="lin_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Linear Regression.

        For more Linear Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1        	

        fit_intercept : boolean, optional, default True
            whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).

        normalize : boolean, optional, default False
            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.

        Returns
        -------
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = LinearRegression

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="bayridge_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Bayesian Ridge Regression model.

        For more Linear Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
        and https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression 

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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

        
            Name of column for labels that are generated, by default "bayridge_reg_predictions"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        
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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = BayesianRidge

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="elastic",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Elastic Net regression with combined L1 and L2 priors as regularizer.
        
        For more Linear Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet 

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold
 
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

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1   
        
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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = ElasticNet

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="lasso",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Lasso Regression Model trained with L1 prior as regularizer (aka the Lasso)

        Technically the Lasso model is optimizing the same objective function as the Elastic Net with l1_ratio=1.0 (no L2 penalty).   

        For more Lasso Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1        
        
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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = Lasso

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="ridge_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Ridge Regression model. 

        For more Ridge Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1        
        
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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = Ridge

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="sgd_reg",
        run=True,
        verbose=1,
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

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1        
        
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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = SGDRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="ada_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains an AdaBoost Regression model.

        An AdaBoost classifier is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset
        but where the weights of incorrectly classified instances are adjusted such that subsequent regressors focus more on difficult cases.

        For more AdaBoost info, you can view it here:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = AdaBoostRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="bag_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Bagging Regressor model.

        A Bagging classifier is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
        Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

        For more Bagging Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = BaggingRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="grad_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Gradient Boosting regression model.

        GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.
        In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. 

        For more Gradient Boosting Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = GradientBoostingRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="rf_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Random Forest Regression model.

        A random forest is a meta estimator that fits a number of decision tree regressors on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
        The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).

        For more Random Forest info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        
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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results
        
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

        model = RandomForestRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="dt_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Decision Tree Regression model.

        For more Decision Tree info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1       	

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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = DecisionTreeRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="linsvr",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Linear Support Vector Regression model.

        Similar to SVR with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm,
        so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.

        For more Support Vector info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1    	

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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = LinearSVR

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="svr_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Epsilon-Support Vector Regression.

        The free parameters in the model are C and epsilon.

        The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples.
        For large datasets consider using model.linearsvr or model.sgd_regression instead

        For more Support Vector info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1    	

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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = SVR

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="xgb_reg",
        run=True,
        verbose=1,
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

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = xgb.XGBRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
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
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="lgbm_reg",
        run=True,
        verbose=1,
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

        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold

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
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results

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

        model = lgb.LGBMRegressor

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model
