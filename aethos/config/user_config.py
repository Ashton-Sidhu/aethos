import os

import yaml
from IPython import get_ipython

from aethos.util import _make_dir

pkg_directory = os.path.dirname(__file__)

with open(
    os.path.join(os.path.expanduser("~"), ".aethos", "config.yml"), "r"
) as ymlfile:
    cfg = yaml.safe_load(ymlfile)

with open(f"{pkg_directory}/technique_reasons.yml", "r") as stream:
    technique_reason_repo = yaml.safe_load(stream)

shell = get_ipython().__class__.__name__

DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".aethos", "models")
DEFAULT_REPORT_DIR = os.path.join(os.path.expanduser("~"), ".aethos", "reports")
DEFAULT_IMAGE_DIR = os.path.join(os.path.expanduser("~"), ".aethos", "images")
DEFAULT_EXPERIMENTS_DIR = "file:" + os.path.join(
    os.path.expanduser("~"), ".aethos", "experiments", "mlruns"
)
DEFAULT_DEPLOYMENTS_DIR = os.path.join(os.path.expanduser("~"), ".aethos", "projects")


def _make_image_dir():

    if not cfg["images"]["dir"]:
        image_dir = DEFAULT_IMAGE_DIR
    else:
        image_dir = cfg["images"]["dir"]

    _make_dir(image_dir)

    return image_dir


IMAGE_DIR = _make_image_dir()


def _make_experiment_dir():  # pragma: no cover

    if not cfg["mlflow"]["dir"]:
        exp_dir = DEFAULT_EXPERIMENTS_DIR
    else:
        exp_dir = cfg["mlflow"]["dir"]

    if exp_dir.startswith("file:"):
        _make_dir(exp_dir[5:])

    return exp_dir


EXP_DIR = _make_experiment_dir()
