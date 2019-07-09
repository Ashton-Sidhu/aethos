import pandas as pd

from categorical import *
from data.data import Data
from data.util import GetListOfCols, _FunctionInputValidation
from numeric import *


class Preprocess():

    
    def __init__(self, data=None, train_data=None, test_data=None, data_properties=None, test_split_percentage=0.8, use_full_data=False, target_field="", reporting=True):        

        if not _FunctionInputValidation(data, train_data, test_data):
            return "Please provide one of either data or train_data and test_data, not both."

        self.data = data

        if data_properties is None:
            self.data_properties = Data(self.data)
            self.data_properties.use_full_data = use_full_data
        else:
            self.data_properties = data_properties

        if self.data is not None:
            self.train_data, self.test_data = self.data_properties.SplitData(test_split_percentage)        
        else:
            self.train_data = self.data_properties.train_data
            self.test_data = self.data_properties.test_data

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

        if self.data_properties.use_full_data:
            self.data = PreprocessNormalize(list_of_cols=list_of_cols, data=self.data)
            
            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data =  PreprocessNormalize(list_of_cols=list_of_cols,
                                                                                                    train_data=self.data_properties.train_data,
                                                                                                    test_data=self.data_properties.test_data)

            return self.data_properties.train_data, self.data_properties.test_data
