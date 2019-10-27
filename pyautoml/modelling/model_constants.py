import sklearn
import xgboost as xgb
from interpret.blackbox import (LimeTabular, MorrisSensitivity,
                                PartialDependence, ShapKernel)
from interpret.perf import PR, ROC, RegressionPerf
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

SHAP_LEARNERS = {
    sklearn.linear_model.LogisticRegression: 'linear',
    sklearn.linear_model.BayesianRidge: 'linear',
    sklearn.linear_model.ElasticNet: 'linear',
    sklearn.linear_model.Lasso: 'linear',
    sklearn.linear_model.LinearRegression: 'linear',
    sklearn.linear_model.Ridge: 'linear',
    sklearn.linear_model.RidgeClassifier: 'linear',
    sklearn.linear_model.SGDClassifier: 'linear',
    sklearn.linear_model.SGDRegressor: 'linear',
    sklearn.ensemble.AdaBoostClassifier: 'kernel',
    sklearn.ensemble.AdaBoostRegressor: 'kernel',
    sklearn.ensemble.BaggingClassifier: 'kernel',
    sklearn.ensemble.BaggingRegressor: 'kernel',
    sklearn.ensemble.GradientBoostingClassifier: 'tree',
    sklearn.ensemble.GradientBoostingRegressor: 'tree',
    sklearn.ensemble.RandomForestClassifier: 'tree',
    sklearn.ensemble.RandomForestRegressor: 'tree',
    BernoulliNB: 'kernel',
    GaussianNB: 'kernel',
    MultinomialNB: 'kernel',
    sklearn.tree.DecisionTreeClassifier: 'tree',
    sklearn.tree.DecisionTreeRegressor: 'tree',
    sklearn.svm.LinearSVC: 'kernel',
    sklearn.svm.LinearSVR: 'kernel',
    sklearn.svm.SVC: 'kernel',
    sklearn.svm.SVR: 'kernel',
    xgb.XGBClassifier: 'tree',
    xgb.XGBRegressor: 'tree',
}

PROBLEM_TYPE = {
    sklearn.linear_model.LogisticRegression: 'classification',
    sklearn.linear_model.BayesianRidge: 'regression',
    sklearn.linear_model.ElasticNet: 'regression',
    sklearn.linear_model.Lasso: 'regression',
    sklearn.linear_model.LinearRegression: 'regression',
    sklearn.linear_model.Ridge: 'regression',
    sklearn.linear_model.RidgeClassifier: 'classification',
    sklearn.linear_model.SGDClassifier: 'classification',
    sklearn.linear_model.SGDRegressor: 'regression',
    sklearn.ensemble.AdaBoostClassifier: 'classification',
    sklearn.ensemble.AdaBoostRegressor: 'regression',
    sklearn.ensemble.BaggingClassifier: 'classification',
    sklearn.ensemble.BaggingRegressor: 'regression',
    sklearn.ensemble.GradientBoostingClassifier: 'classification',
    sklearn.ensemble.GradientBoostingRegressor: 'regression',
    sklearn.ensemble.RandomForestClassifier: 'classification',
    sklearn.ensemble.RandomForestRegressor: 'regression',
    sklearn.naive_bayes.BernoulliNB: 'classification',
    sklearn.naive_bayes.GaussianNB: 'classification',
    sklearn.naive_bayes.MultinomialNB: 'classification',
    sklearn.tree.DecisionTreeClassifier: 'classification',
    sklearn.tree.DecisionTreeRegressor: 'regression',
    sklearn.svm.LinearSVC: 'classification',
    sklearn.svm.LinearSVR: 'regression',
    sklearn.svm.SVC: 'classification',
    sklearn.svm.SVR: 'regression',
    xgb.XGBClassifier: 'classification',
    xgb.XGBRegressor: 'regression',     
}

INTERPRET_EXPLAINERS = {
    'problem': {
        'classification': {
            'roc': ROC,
            'pr' : PR,
        },
        'regression': {
            'regperf': RegressionPerf,
        },
    },    
    'local': {
        'lime': LimeTabular,
        'shap': ShapKernel,
    },
    'global': {
        'morris': MorrisSensitivity,
        'dependence': PartialDependence,
    }
}
