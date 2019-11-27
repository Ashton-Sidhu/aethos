import os
import sys

import interpret
import pandas as pd
# let init-time option registration happen
import pyautoml.config.config_init
import shap
from bokeh.io import output_notebook
from IPython import get_ipython
from pyautoml.config.config import (describe_option, get_option, options,
                                    reset_option, set_option)

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
