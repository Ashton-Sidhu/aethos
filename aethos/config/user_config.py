import os
import yaml
from IPython import get_ipython

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
