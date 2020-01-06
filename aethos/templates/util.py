import os

from aethos.config import DEFAULT_DEPLOYMENTS_DIR, cfg
from aethos.util import _make_dir


def _create_dir():
    """
    Creates the projects directory.
    
    Parameters
    ----------
    project_dir : str
        Full path of the project dir.

    name : str
        Name of the project
    """

    if not cfg["models"]["deployment_dir"]:
        dep_dir = DEFAULT_DEPLOYMENTS_DIR
    else:
        dep_dir = cfg["models"]["deployment_dir"]

    _make_dir(dep_dir)

    return dep_dir

def _create_project_dir(project_dir: str, name: str):
    """
    Creates the projects directory.
    
    Parameters
    ----------
    project_dir : str
        Full path of the project dir.

    name : str
        Name of the project
    """

    project_dir = os.path.join(project_dir, name)

    _make_dir(project_dir)

    return project_dir

def _get_model_type_kwarg(model):
    """
    Helper function to determine what model to import in requirements.txt service file.
    """

    import xgboost as xgb
    import catboost as cb
    import lightgbm as lgb

    kwargs = {
        'xgboost': False,
        'catboost': False,
        'lgbm': False,
    }

    if isinstance(model, xgb.XGBModel):
        kwargs['xgboost'] = True
    
    if isinstance(model, cb.CatBoost):
        kwargs['catboost'] = True

    if isinstance(model, lgb.LGBMModel):
        kwargs['lgbm'] = True

    return kwargs
