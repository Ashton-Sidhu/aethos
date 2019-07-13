import pandas as pd
import yaml

from pyautoml.data.data import Data
from pyautoml.preprocessing.categorical import *
from pyautoml.preprocessing.numeric import *
from pyautoml.preprocessing.text import *
from pyautoml.util import GetListOfCols, SplitData, _FunctionInputValidation

with open("pyautoml/technique_reasons.yml", 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")

class Preprocess():

    
    def __init__(self, data=None, train_data=None, test_data=None, data_properties=None, test_split_percentage=0.8, use_full_data=False, target_field="", report_name=None):        

        if not _FunctionInputValidation(data, train_data, test_data):
            print("Error initialzing constructor, please provide one of either data or train_data and test_data, not both.")

        self.data = data

        if data_properties is None:
            self.data_properties = Data(self.data)
            self.data_properties.use_full_data = use_full_data
        else:
            self.data_properties = data_properties

        if self.data is not None:
            self.train_data, self.test_data = SplitData(self.data, test_split_percentage)        
        else:
            self.train_data = self.data_properties.train_data
            self.test_data = self.data_properties.test_data

        if self.data_properties.report is None:
            self.report = None
        else:
            self.report = self.data_properties.report
            self.report.WriteHeader("Preprocessing")

    def NormalizeNumeric(self, list_of_cols=[]):
        """Function that normalizes all numeric values between 0 and 1 to bring features into same domain.

       This function can be found in `preprocess/numeric.py`
        
        If `list_of_cols` is not provided, the strategy will be applied to all numeric columns.
        
        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
        
        Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned. 
        """

        report_info = technique_reason_repo['preprocess']['numeric']['standardize']

        if self.data_properties.use_full_data:
            self.data = PreprocessNormalize(list_of_cols=list_of_cols, data=self.data)

            if self.report is not None:
                self.report.ReportTechnique(report_info, list_of_cols)
            
            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = PreprocessNormalize(list_of_cols=list_of_cols,
                                                                                                    train_data=self.data_properties.train_data,
                                                                                                    test_data=self.data_properties.test_data)

            if self.report is not None:
                self.report.ReportTechnique(report_info, list_of_cols)

            return self.data_properties.train_data, self.data_properties.test_data
