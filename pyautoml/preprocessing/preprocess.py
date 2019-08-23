import os

import pandas as pd
import pyautoml
import yaml
from pyautoml.base import MethodBase
from pyautoml.preprocessing.categorical import *
from pyautoml.preprocessing.numeric import *
from pyautoml.preprocessing.text import *
from pyautoml.util import _input_columns, _numeric_input_conditions

pkg_directory = os.path.dirname(pyautoml.__file__)

with open(f"{pkg_directory}/technique_reasons.yml", 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")

class Preprocess(MethodBase):

    
    def __init__(self, data=None, train_data=None, test_data=None, data_properties=None, test_split_percentage=0.2, use_full_data=False, target_field="", report_name=None):

        if data_properties is None:        
            super().__init__(data=data, train_data=train_data, test_data=test_data, test_split_percentange=test_split_percentage,
                        use_full_data=use_full_data, target_field=target_field, report_name=report_name)
        else:
            super().__init__(data=data_properties.data, train_data=data_properties.train_data, test_data=data_properties.test_data, test_split_percentange=test_split_percentage,
                        use_full_data=data_properties.use_full_data, target_field=data_properties.target_field, report_name=data_properties.report.filename)
                        

        if self.data_properties.report is not None:
            self.report.write_header("Preprocessing")

        
    def normalize_numeric(self, *list_args, list_of_cols=[], normalize_params={}):
        """
        Function that normalizes all numeric values between 0 and 1 to bring features into same domain.
        
        If `list_of_cols` is not provided, the strategy will be applied to all numeric columns.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function can be found in `preprocess/numeric.py`     
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        normalize_params : dict, optional
            A dictionary of parmaters to pass into MinMaxScaler() constructor
            from Scikit-Learn, by default {}
        
        Returns
        -------
        Dataframe:
            Top 10 rows of data or the training data to view analysis.
        """

        report_info = technique_reason_repo['preprocess']['numeric']['standardize']
        
        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if self.data_properties.use_full_data:
            self.data_properties.data = preprocess_normalize(list_of_cols=list_of_cols, params=normalize_params, data=self.data_properties.data)

            if self.report is not None:
                if list_of_cols:
                    self.report.report_technique(report_info, list_of_cols)
                else:
                    list_of_cols = _numeric_input_conditions(list_of_cols, self.data_properties.data, None)
                    self.report.report_technique(report_info, list_of_cols)
            
            return self.data_properties.data.head(10)

        else:
            self.data_properties.train_data, self.data_properties.test_data = preprocess_normalize(list_of_cols=list_of_cols,
                                                                                                    params=normalize_params,
                                                                                                    train_data=self.data_properties.train_data,
                                                                                                    test_data=self.data_properties.test_data)

            if self.report is not None:
                if list_of_cols:
                    self.report.report_technique(report_info, list_of_cols)
                else:
                    list_of_cols = _numeric_input_conditions(list_of_cols, None, self.data_properties.train_data)
                    self.report.report_technique(report_info, list_of_cols)

            return self.data_properties.train_data.head(10)
