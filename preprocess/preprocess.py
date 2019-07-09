import pandas as pd

from categorical import *
from data.data import Data
from data.util import GetListOfCols, _FunctionInputValidation
from numeric import *


class Preprocess():

    
    def __init__(self, data=None, train_data=None, test_data=None, data_properties=None, test_split_percentage=0.2, target_field="", reporting=True):        

        if not _FunctionInputValidation(data, train_data, test_data):
            return "Please provide one of either data or train_data and test_data, not both."

        self.data = data

        if data_properties is None:
            self.data_properties = Data(self.data)
        else:
            self.data_properties = data_properties

        if self.data is None:
            self.train_data, self.test_data = self.data_properties.SplitData(test_split_percentage)
        
        else:
            self.train_data = self.data_properties.train_data
            self.test_data = self.data_properties.test_data

    def NormalizeNumeric(self, list_of_cols=[]):

        if self.data is not None:
            self.data = PreprocessNormalize(list_of_cols=list_of_cols, data=self.data)
            
            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data =  PreprocessNormalize(list_of_cols=list_of_cols,
                                                                                                    train_data=self.data_properties.train_data,
                                                                                                    test_data=self.data_properties.test_data)

            return self.data_properties.train_data, self.data_properties.test_data
