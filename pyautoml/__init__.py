import os

import interpret
import pandas as pd
import shap
import yaml
from bokeh.io import output_notebook
from IPython import get_ipython

import pyautoml

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
    shap.initjs()

pkg_directory = os.path.dirname(pyautoml.__file__)

with open("{}/technique_reasons.yml".format(pkg_directory), 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")
