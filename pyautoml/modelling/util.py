import inspect
import multiprocessing as mp
import warnings
from functools import partial

from pathos.multiprocessing import ProcessingPool
from sklearn.model_selection import GridSearchCV


def add_to_queue(model_function):
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

def run_gridsearch(model, gridsearch, grid_params, cv: int, score: str):
    """
    Runs Gridsearch on a model
    
    Parameters
    ----------
    model : Model
        Model to run gridsearch on

    gridsearch : bool or dist
        True, False or custom grid to test

    grid_params : dict
        Dictionary of model params and values to test

    cv : int
        Number of folds for cross validation
    
    score : str
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

    model = GridSearchCV(model, gridsearch_grid, cv=cv, scoring=score)

    return model

def _run_models(model_obj):

    p = ProcessingPool(mp.cpu_count())

    results = p.map(_run, list(model_obj._queued_models.values()))

    for result in results:
        model_obj._models[result.model_name] = result

    p.close()
    p.join()


def _run(model):
    return model()
