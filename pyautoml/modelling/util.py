import inspect
import warnings
from functools import partial

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
            warnings.warn("Running models all at once not available yet. Please set `run=True` to train your model.")
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

def run_gridsearch(model, gridsearch, grid_params, cv, score):

    if isinstance(gridsearch, dict):
        gridsearch_grid = gridsearch
    elif gridsearch == True:
        gridsearch_grid = grid_params
        print("Gridsearching with the following parameters: {}".format(grid_params))
    else:
        raise ValueError("Invalid Gridsearch input.")

    model = GridSearchCV(model, gridsearch_grid, cv=cv, scoring=score)

    return model
