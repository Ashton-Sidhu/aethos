import pandas as pd
from bokeh.io import output_notebook
from IPython import get_ipython

from .cleaning import Clean
from .feature_engineering import Feature
from .modelling import Model
from .preprocessing import Preprocess

pd.options.mode.chained_assignment = None


__all__ = ['Clean',
         'Feature',
         'Model',
         'Preprocess'
        ]

shell = get_ipython().__class__.__name__

if shell == 'ZMQInteractiveShell':
    output_notebook()
