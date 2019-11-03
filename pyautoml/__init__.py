import interpret
import pandas as pd
import shap
from bokeh.io import output_notebook
from IPython import get_ipython

from pyautoml.config.config import (
    get_option,
    set_option,
    reset_option,
    describe_option,
    options,
)

# let init-time option registration happen
import pyautoml.config.config_init

from .cleaning import Clean
from .feature_engineering import Feature
from .modelling import Model
from .preprocessing import Preprocess

pd.options.mode.chained_assignment = None

__all__ = ["Clean", "Feature", "Model", "Preprocess"]

shell = get_ipython().__class__.__name__

if shell == "ZMQInteractiveShell":
    output_notebook()
    shap.initjs()
