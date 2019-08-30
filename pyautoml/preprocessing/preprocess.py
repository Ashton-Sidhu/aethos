import copy
import os

import pandas as pd
import pyautoml
import yaml
from pyautoml.base import MethodBase
from pyautoml.preprocessing.categorical import *
from pyautoml.preprocessing.numeric import *
from pyautoml.preprocessing.text import *
from pyautoml.util import (_contructor_data_properties, _input_columns,
                           _numeric_input_conditions)

pkg_directory = os.path.dirname(pyautoml.__file__)

with open(f"{pkg_directory}/technique_reasons.yml", 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")

class Preprocess(MethodBase):

    
    def __init__(self, step=None, data=None, train_data=None, test_data=None, test_split_percentage=0.2, split=True, target_field="", report_name=None):

        _data_properties = _contructor_data_properties(step)

        if _data_properties is None:        
            super().__init__(data=data, train_data=train_data, test_data=test_data, test_split_percentage=test_split_percentage,
                        split=split, target_field=target_field, report_name=report_name)
        else:
            super().__init__(data=_data_properties.data, train_data=_data_properties.train_data, test_data=_data_properties.test_data, test_split_percentage=test_split_percentage,
                        split=_data_properties.split, target_field=_data_properties.target_field, report_name=_data_properties.report_name)
                        

        if self._data_properties.report is not None:
            self.report.write_header("Preprocessing")

        
    def normalize_numeric(self, *list_args, list_of_cols=[], **normalize_params):
        """
        Function that normalizes all numeric values between 0 and 1 to bring features into same domain.
        
        If `list_of_cols` is not provided, the strategy will be applied to all numeric columns.

        If a list of columns is provided use the list, otherwise use arguments.

        This function can be found in `preprocess/numeric.py`     
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        normalize_params : dict, optional
            Parmaters to pass into MinMaxScaler() constructor
            from Scikit-Learn, by default {}
        
        Returns
        -------
        Preprocess Object:
            Returns a deep copy of the Preprocess object.
        """

        report_info = technique_reason_repo['preprocess']['numeric']['standardize']
        
        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = preprocess_normalize(list_of_cols=list_of_cols, **normalize_params, data=self._data_properties.data)

            if self.report is not None:
                if list_of_cols:
                    self.report.report_technique(report_info, list_of_cols)
                else:
                    list_of_cols = _numeric_input_conditions(list_of_cols, self._data_properties.data, None)
                    self.report.report_technique(report_info, list_of_cols)
            
            return self.copy()

        else:
            self._data_properties.train_data, self._data_properties.test_data = preprocess_normalize(list_of_cols=list_of_cols,
                                                                                                    **normalize_params,
                                                                                                    train_data=self._data_properties.train_data,
                                                                                                    test_data=self._data_properties.test_data)

            if self.report is not None:
                if list_of_cols:
                    self.report.report_technique(report_info, list_of_cols)
                else:
                    list_of_cols = _numeric_input_conditions(list_of_cols, None, self._data_properties.train_data)
                    self.report.report_technique(report_info, list_of_cols)

            return self.copy()

    def sentence_split(self, *list_args, list_of_cols=[]):
        """
        Splits text data into sentences and saves it into another column for analysis.

        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        Returns
        -------
        Preprocess Object:
            Returns a deep copy of the Preprocess object.
        """

        report_info = technique_reason_repo['preprocess']['text']['split_sentence']

        list_of_cols = _input_columns(list_args, list_of_cols)        
    
        if not self._data_properties.split:    
            self._data_properties.data = split_sentence(list_of_cols, data=self._data_properties.data)
    
            if self.report is not None:
                self.report.ReportTechnique(report_info)
    
            return self.copy()
    
        else:
            self._data_properties.train_data, self._data_properties.test_data = split_sentence(list_of_cols, 
                                                                                            train_data=self._data_properties.train_data, 
                                                                                            test_data=self._data_properties.test_data)
    
            if self.report is not None:
                self.report.ReportTechnique(report_info)
    
            return self.copy()
