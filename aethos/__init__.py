import warnings

warnings.simplefilter("ignore", FutureWarning)

import pandas as pd

# let init-time option registration happen
import aethos.config.config_init
import shap
from bokeh.io import output_notebook
from IPython import get_ipython
from aethos.config.config import (
    describe_option,
    get_option,
    options,
    reset_option,
    set_option,
)

from .core import Data
from .modelling import Model

pd.options.mode.chained_assignment = None

__all__ = ["Data", "Model"]

shell = get_ipython().__class__.__name__

if shell == "ZMQInteractiveShell":
    output_notebook()
    shap.initjs()
