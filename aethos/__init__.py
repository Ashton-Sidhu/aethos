import warnings

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)

import pandas as pd
import shap
from bokeh.io import output_notebook
from IPython import get_ipython
import plotly.io as pio

# let init-time option registration happen
import aethos.config.config_init
from aethos.config.config import (describe_option, get_option, options,
                                  reset_option, set_option)

from .core import Data
from .modelling import Model

pd.options.mode.chained_assignment = None
pio.templates.default = "plotly_white"

__all__ = ["Data", "Model"]

shell = get_ipython().__class__.__name__

if shell == "ZMQInteractiveShell":
    output_notebook()
    shap.initjs()
