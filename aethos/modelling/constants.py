import catboost as cb
import lightgbm as lgb
import sklearn
import xgboost as xgb
from interpret.blackbox import (
    LimeTabular,
    MorrisSensitivity,
    PartialDependence,
    ShapKernel,
)
from interpret.perf import PR, ROC, RegressionPerf
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

SHAP_LEARNERS = {
    sklearn.linear_model.LogisticRegression: "linear",
    sklearn.linear_model.BayesianRidge: "linear",
    sklearn.linear_model.ElasticNet: "linear",
    sklearn.linear_model.Lasso: "linear",
    sklearn.linear_model.LinearRegression: "linear",
    sklearn.linear_model.Ridge: "linear",
    sklearn.linear_model.RidgeClassifier: "linear",
    sklearn.linear_model.SGDClassifier: "linear",
    sklearn.linear_model.SGDRegressor: "linear",
    sklearn.ensemble.AdaBoostClassifier: "kernel",
    sklearn.ensemble.AdaBoostRegressor: "kernel",
    sklearn.ensemble.BaggingClassifier: "kernel",
    sklearn.ensemble.BaggingRegressor: "kernel",
    sklearn.ensemble.GradientBoostingClassifier: "tree",
    sklearn.ensemble.GradientBoostingRegressor: "tree",
    sklearn.ensemble.RandomForestClassifier: "tree",
    sklearn.ensemble.RandomForestRegressor: "tree",
    BernoulliNB: "kernel",
    GaussianNB: "kernel",
    MultinomialNB: "kernel",
    sklearn.tree.DecisionTreeClassifier: "tree",
    sklearn.tree.DecisionTreeRegressor: "tree",
    sklearn.svm.LinearSVC: "kernel",
    sklearn.svm.LinearSVR: "kernel",
    sklearn.svm.SVC: "kernel",
    sklearn.svm.SVR: "kernel",
    xgb.XGBClassifier: "tree",
    xgb.XGBRegressor: "tree",
    lgb.sklearn.LGBMClassifier: "tree",
    lgb.sklearn.LGBMRegressor: "tree",
    cb.CatBoostRegressor: "tree",
    cb.CatBoostClassifier: "tree",
}

PROBLEM_TYPE = {
    sklearn.linear_model.LogisticRegression: "classification",
    sklearn.linear_model.BayesianRidge: "regression",
    sklearn.linear_model.ElasticNet: "regression",
    sklearn.linear_model.Lasso: "regression",
    sklearn.linear_model.LinearRegression: "regression",
    sklearn.linear_model.Ridge: "regression",
    sklearn.linear_model.RidgeClassifier: "classification",
    sklearn.linear_model.SGDClassifier: "classification",
    sklearn.linear_model.SGDRegressor: "regression",
    sklearn.ensemble.AdaBoostClassifier: "classification",
    sklearn.ensemble.AdaBoostRegressor: "regression",
    sklearn.ensemble.BaggingClassifier: "classification",
    sklearn.ensemble.BaggingRegressor: "regression",
    sklearn.ensemble.GradientBoostingClassifier: "classification",
    sklearn.ensemble.GradientBoostingRegressor: "regression",
    sklearn.ensemble.RandomForestClassifier: "classification",
    sklearn.ensemble.RandomForestRegressor: "regression",
    sklearn.naive_bayes.BernoulliNB: "classification",
    sklearn.naive_bayes.GaussianNB: "classification",
    sklearn.naive_bayes.MultinomialNB: "classification",
    sklearn.tree.DecisionTreeClassifier: "classification",
    sklearn.tree.DecisionTreeRegressor: "regression",
    sklearn.svm.LinearSVC: "classification",
    sklearn.svm.LinearSVR: "regression",
    sklearn.svm.SVC: "classification",
    sklearn.svm.SVR: "regression",
    xgb.XGBClassifier: "classification",
    xgb.XGBRegressor: "regression",
    lgb.sklearn.LGBMClassifier: "classification",
    lgb.sklearn.LGBMRegressor: "regression",
    cb.CatBoostRegressor: "regression",
    cb.CatBoostClassifier: "classification",
}

INTERPRET_EXPLAINERS = {
    "problem": {
        "classification": {"roc": ROC, "pr": PR},
        "regression": {"regperf": RegressionPerf},
    },
    "local": {"lime": LimeTabular, "shap": ShapKernel},
    "global": {"morris": MorrisSensitivity, "dependence": PartialDependence},
}

CLASS_METRICS_DESC = {
    "Accuracy": "Measures how many observations, both positive and negative, were correctly classified.",
    "Balanced Accuracy": "The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.",
    "Average Precision": "Summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold",
    "ROC AUC": "Shows how good at ranking predictions your model is. It tells you what is the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.",
    "Zero One Loss": "Fraction of misclassifications.",
    "Precision": "It measures how many observations predicted as positive are positive. Good to use when False Positives are costly.",
    "Recall": "It measures how many observations out of all positive observations have we classified as positive. Good to use when catching call positive occurences, usually at the cost of false positive.",
    "Matthews Correlation Coefficient": "It’s a correlation between predicted classes and ground truth.",
    "Log Loss": "Difference between ground truth and predicted score for every observation and average those errors over all observations.",
    "Jaccard": "Defined as the size of the intersection divided by the size of the union of two label sets, is used to compare set of predicted labels for a sample to the corresponding set of true labels.",
    "Hinge Loss": "Computes the average distance between the model and the data using hinge loss, a one-sided metric that considers only prediction errors.",
    "Hamming Loss": "The Hamming loss is the fraction of labels that are incorrectly predicted.",
    "F-Beta": "It’s the harmonic mean between precision and recall, with an emphasis on one or the other. Takes into account both metrics, good for imbalanced problems (spam, fraud, etc.).",
    "F1": "It’s the harmonic mean between precision and recall. Takes into account both metrics, good for imbalanced problems (spam, fraud, etc.).",
    "Cohen Kappa": "Cohen Kappa tells you how much better is your model over the random classifier that predicts based on class frequencies. Works well for imbalanced problems.",
    "Brier Loss": "It is a measure of how far your predictions lie from the true values. Basically, it is a mean square error in the probability space.",
}

REG_METRICS_DESC = {
    "Explained Variance": "Explained variance regression score function. Best possible score is 1.0, lower values are worse.",
    "Max Error": "Returns the single most maximum residual error.",
    "Mean Absolute Error": "Postive mean value of all residuals",
    "Mean Squared Error": "Mean of the squared sum the residuals",
    "Root Mean Sqaured Error": "Square root of the Mean Squared Error",
    "Mean Squared Log Error": "Mean of the squared sum of the log of all residuals",
    "Median Absolute Error": "Postive median value of all residuals",
    "R2": "R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable that is explained by an independent variable or variables in a regression model.",
    "SMAPE": "Symmetric mean absolute percentage error. It is an accuracy measure based on percentage (or relative) errors.",
}

DEBUG_OVERFIT = [
    "Add data to your training set.",
    "Add or increase regularization. (L2 regularization, L1 regularization, dropout)",
    "Add early stopping.",
    "Feature selection to decrease number/type of input features.",
    "Modify input features based on insights from error analysis​ - add more features.",
    "Modify model architecture",
]

DEBUG_UNDERFIT = [
    "Increase the size of your model (for example, increase thesize of your neural network by adding layers/neurons)",
    "Modify input features based on insights from error analysis​ - add more features",
    "Reduce or eliminate regularization​.",
]
