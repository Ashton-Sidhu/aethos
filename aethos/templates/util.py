import os


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

    os.system(f"mkdir -p {project_dir}/{name}/app")

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
