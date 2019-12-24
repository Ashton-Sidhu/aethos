xgb_reg_doc = """
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
"""


lgbm_reg_doc = """
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
"""


catboost_reg_doc = """
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
"""
