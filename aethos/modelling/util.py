import inspect
import multiprocessing as mp
import os
import pickle
import warnings
from functools import partial, wraps
from pathlib import Path

import matplotlib.pyplot as plt
from aethos.config import DEFAULT_MODEL_DIR, cfg
from aethos.util import _make_dir
from aethos.visualizations.util import _make_image_dir
from pathos.multiprocessing import ProcessingPool
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from yellowbrick.model_selection import CVScores, LearningCurve


def add_to_queue(model_function):
    @wraps(model_function)
    def wrapper(self, *args, **kwargs):
        default_kwargs = get_default_args(model_function)

        kwargs["run"] = kwargs.get("run", default_kwargs["run"])
        kwargs["model_name"] = kwargs.get("model_name", default_kwargs["model_name"])
        cv = kwargs.get("cv", False)

        if not _validate_model_name(self, kwargs["model_name"]):
            raise AttributeError("Invalid model name. Please choose another one.")

        if kwargs["run"] or cv:
            return model_function(self, *args, **kwargs)
        else:
            kwargs["run"] = True
            self._queued_models[kwargs["model_name"]] = partial(
                model_function, self, *args, **kwargs
            )

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


def run_gridsearch(model, gridsearch, cv=5, scoring="accuracy", **gridsearch_kwargs):
    """
    Runs Gridsearch on a model
    
    Parameters
    ----------
    model : Model
        Model to run gridsearch on

    gridsearch : dict
        Dict of params to test

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
        print(f"Gridsearching with the following parameters: {gridsearch_grid}")
    else:
        raise ValueError("Invalid Gridsearch input.")

    model = GridSearchCV(
        model, gridsearch_grid, cv=cv, scoring=scoring, **gridsearch_kwargs
    )

    return model


def run_crossvalidation(
    model, x_train, y_train, cv=5, scoring="accuracy", report=None, model_name=None
):
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
        Cross validation method, by default 5

    scoring : str, optional
        Scoring method, by default 'accuracy'
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    visualizer_scores = CVScores(model, cv=cv, scoring=scoring, ax=axes[0])
    visualizer_scores.fit(x_train, y_train)
    visualizer_scores.finalize()

    visualizer_lcurve = LearningCurve(model, cv=cv, scoring=scoring, ax=axes[1])
    visualizer_lcurve.fit(x_train, y_train)
    visualizer_lcurve.finalize()

    visualizer_scores.show()
    visualizer_lcurve.show()

    if report:  # pragma: no cover
        imgdir = _make_image_dir()
        fig.savefig(os.path.join(imgdir, model_name + ".svg"))


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
    """
    Runs a model
        
    Returns
    -------
    Model
        Trained model
    """
    return model()


def _get_cv_type(cv_type, random_state, **kwargs):
    """
    Takes in cv type from the user and initiates the cross validation generator.
    
    Parameters
    ----------
    cv_type : int, str or None
        Crossvalidation type

    random_state : int
        Random seed
    
    Returns
    -------
    Cross Validation Generator
        CV Generator
    """

    if not cv_type:
        return None, kwargs

    if isinstance(cv_type, int):
        cv_type = cv_type
    elif cv_type == "kfold":
        n_splits = kwargs.pop("n_splits", 5)
        shuffle = kwargs.pop("shuffle", False)

        cv_type = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    elif cv_type == "strat-kfold":
        n_splits = kwargs.pop("n_splits", 5)
        shuffle = kwargs.pop("shuffle", False)

        cv_type = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
    else:
        raise ValueError("Cross Validation type is invalid.")

    return cv_type, kwargs


def to_pickle(model, name, project=False, project_name=None):
    """
    Writes model to a pickle file.
    
    Parameters
    ----------
    model: Model object
        Model object to serialize

    name : str
        Name of the model
        
    project : bool
        Whether to write to the project folder, by default False
    """

    if not project:
        if not cfg["models"]["dir"]:  # pragma: no cover
            path = DEFAULT_MODEL_DIR
        else:
            path = cfg["models"]["dir"]
    else:
        path = os.path.join(
            os.path.expanduser("~"), ".aethos", "projects", project_name, "app"
        )

    if not os.path.exists(path):
        os.makedirs(path)

    pickle.dump(model, open(os.path.join(path, name + ".pkl"), "wb"))


def _validate_model_name(model_obj, model_name: str) -> bool:
    """
    Validates the inputted model name. If the object already has an
    attribute with that model name, it is invalid
    
    Parameters
    ----------
    model_name : str
        Proposed name of the model
        
    model_obj : Model
        Model object
    
    Returns
    -------
    bool
        True if model name is valid, false otherwise
    """

    if hasattr(model_obj, model_name) and model_name not in model_obj._models:
        return False

    return True
