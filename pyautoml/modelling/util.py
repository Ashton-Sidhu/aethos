import inspect
import warnings
from functools import partial


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
