import pandas as pd

from categorical import *
from data.data import Data
from data.util import GetListOfCols, _FunctionInputValidation
from numeric import *
from text import *


class Feature():

    
    def __init__(self, data=None, train_data=None, test_data=None, data_properties=None, test_split_percentage=0.2, use_full_data=False, target_field="", reporting=True):        

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

    def OneHotEncode(self, list_of_cols, data=None, train_data=None, test_data=None):
        """Creates a matrix of converted categorical columns into binary columns.
    
        Either data or train_data or test_data MUST be provided, not both. 

        Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
        
        Keyword Arguments:            
            data {DataFrame} -- Full dataset (default: {None})
            train_data {DataFrame} -- Training dataset (default: {None})
            test_data {DataFrame} -- Testing dataset (default: {None})
            **tfidf_kwargs {dictionary} - Parameters you would pass into Bag of Words constructor as a dictionary

        Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned.  
        
        """

        if self.data_properties.use_full_data:

            self.data = FeatureOneHotEncode(list_of_cols, data=self.data)

            return self.data

        else:

            self.data_properties.train_data, self.data_properties.test_data = FeatureOneHotEncode(list_of_cols,
                                                                                                train_data=self.data_properties.train_data,
                                                                                                test_data=self.data_properties.test_data)

            return self.data_properties.train_data, self.data_properties.test_data


    def TFIDF(self, list_of_cols=[], params={}):
        """Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.

        This function exists in `feature-extraction/text.py`
        
        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
            tfidf_kwargs {dictionary} - Parameters you would pass into Bag of Words constructor as a dictionary
        
        Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned. 
        """

        if self.data_properties.use_full_data:

            self.data = FeatureTFIDF(list_of_cols=list_of_cols, data=self.data, params=params) 

            return self.data

        else:

            self.data_properties.train_data, self.data_properties.test_data = FeatureTFIDF(list_of_cols=list_of_cols,
                                                                                            train_data=self.data_properties.train_data,
                                                                                            test_data=self.data_properties.test_data,
                                                                                            params=params)

            return self.data_properties.train_data, self.data_properties.test_data

    def BagofWords(self, list_of_cols=[], params={}):
        """Creates a matrix of how many times a word appears in a document.

        This function exists in `feature-extraction/text.py`

        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
            params {dictionary} - Parameters you would pass into Bag of Words constructor as a dictionary

        Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned. 
        """

        if self.data_properties.use_full_data:

            self.data = FeatureBagOfWords(list_of_cols, data=self.data, params=params) 

            return self.data

        else:

            self.data_properties.train_data, self.data_properties.test_data = FeatureBagOfWords(list_of_cols,
                                                                                            train_data=self.data_properties.train_data,
                                                                                            test_data=self.data_properties.test_data,
                                                                                            params=params)

            return self.data_properties.train_data, self.data_properties.test_data
