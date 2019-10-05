import inspect
import multiprocessing as mp
import warnings
from functools import partial, wraps

from pathos.multiprocessing import ProcessingPool
from sklearn.model_selection import GridSearchCV

from yellowbrick.model_selection import CVScores, LearningCurve


def add_to_queue(model_function):
    @wraps(model_function)
    def wrapper(self, *args, **kwargs):
        default_kwargs = get_default_args(model_function)

        if 'run' not in kwargs:
            kwargs['run'] = default_kwargs['run']
        if 'model_name' not in kwargs:
            kwargs['model_name'] = default_kwargs['model_name']

        if kwargs['run']:
            return model_function(self, *args, **kwargs)
        else:
            kwargs['run'] = True
            self._queued_models[kwargs['model_name']] = partial(model_function, self, *args, **kwargs)
    
    return wrapper

def get_default_args(func):
    """
    Gets default arguments from a function.
    """

    signature = inspect.signature(func)

    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

# TODO: add ability to take in cv objects
def run_gridsearch(model, gridsearch, grid_params, cv=12, scoring='accuracy', **gridsearch_kwargs):
    """
    Runs Gridsearch on a model
    
    Parameters
    ----------
    model : Model
        Model to run gridsearch on

    gridsearch : bool or dict
        True, False or custom grid to test

    grid_params : dict
        Dictionary of model params and values to test

    cv : int, Crossvalidation Generator, optional
        Cross validation method, by default 12
    
    scoring : str
        Scoring metric to use when evaluating models
    
    Returns
    -------
    Model
        Initialized Gridsearch model
    """

    if isinstance(gridsearch, dict):
        gridsearch_grid = gridsearch
    elif gridsearch == True:
        gridsearch_grid = grid_params
        print("Gridsearching with the following parameters: {}".format(grid_params))
    else:
        raise ValueError("Invalid Gridsearch input.")

    model = GridSearchCV(model, gridsearch_grid, cv=cv, scoring=scoring, **gridsearch_kwargs)
    
    return model

def run_crossvalidation(model, x_train, y_train, cv=12, scoring='accuracy', learning_curve=False, **kwargs):
    """
    Runs cross validation on a certain model.
    
    Parameters
    ----------
    model : Model
        Model to cross validate

    x_train : nd-array
        Training data

    y_train : nd-array
        Testing data

    cv : int, Crossvalidation Generator, optional
        Cross validation method, by default 12

    scoring : str, optional
        Scoring method, by default 'accuracy'

    learning_curve : bool, optional
        If true plot learning curve, by default False
    
    Returns
    -------
    list
        List of cross validation curves
    """

    if isinstance(cv, int):
        cv = cv
    else:
        cv = cv(**kwargs)

    visualizer_scores = CVScores(model, cv=cv, scoring=scoring)
    visualizer_scores.fit(x_train, y_train)
    visualizer_scores.poof()

    if learning_curve:
        visualizer_lcurve = LearningCurve(model, cv=cv, scoring=scoring)
        visualizer_lcurve.fit(x_train, y_train)
        visualizer_lcurve.poof()

    return visualizer_scores.cv_scores_

def _run_models_parallel(model_obj):
    """
    Runs queued models in parallel
    
    Parameters
    ----------
    model_obj : Model
        Model object
    """

    p = ProcessingPool(mp.cpu_count())

    results = p.map(_run, list(model_obj._queued_models.values()))

    for result in results:
        model_obj._models[result.model_name] = result

    p.close()
    p.join()

def _run(model):
    return model()
