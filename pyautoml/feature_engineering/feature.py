import os

import pandas as pd
import pyautoml
import yaml
from pyautoml.data.data import Data
from pyautoml.feature_engineering.categorical import *
from pyautoml.feature_engineering.numeric import *
from pyautoml.feature_engineering.text import *
from pyautoml.util import GetListOfCols, SplitData, _FunctionInputValidation

pkg_directory = os.path.dirname(pyautoml.__file__)

with open(f"{pkg_directory}/technique_reasons.yml", 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")


class Feature():

    def __init__(self, data=None, train_data=None, test_data=None, data_properties=None, test_split_percentage=0.2, use_full_data=False, target_field="", report_name=None):

        if not _FunctionInputValidation(data, train_data, test_data):
            print("Error initialzing constructor, please provide one of either data or train_data and test_data, not both.")

        if data_properties is None:
            self.data_properties = Data(
                data, train_data, test_data, use_full_data=use_full_data, target_field=target_field, report_name=report_name)

            if data is not None:
                self.data = data
                self.data_properties.train_data, self.data_properties.test_data = SplitData(
                    self.data, test_split_percentage)

        else:
            self.data = data
            self.data_properties = data_properties

        if self.data_properties.report is None:
            self.report = None
        else:
            self.report = self.data_properties.report
            self.report.WriteHeader("Feature Engineering")

        self.train_data = self.data_properties.train_data
        self.test_data = self.data_properties.test_data

    def onehot_encode(self, list_of_cols, data=None, train_data=None, test_data=None):
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
        report_info = technique_reason_repo['feature']['categorical']['onehotencode']

        if self.data_properties.use_full_data:

            self.data = FeatureOneHotEncode(list_of_cols, data=self.data)

            if self.report is not None:
                self.report.ReportTechnique(report_info, list_of_cols)

            return self.data

        else:

            self.data_properties.train_data, self.data_properties.test_data = FeatureOneHotEncode(list_of_cols,
                                                                                                  train_data=self.data_properties.train_data,
                                                                                                  test_data=self.data_properties.test_data)
            if self.report is not None:
                self.report.ReportTechnique(report_info, list_of_cols)

            return self.data_properties.train_data, self.data_properties.test_data

    def tfidf(self, list_of_cols=[], params={}):
        """Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.

        This function exists in `feature-extraction/text.py`

        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
            tfidf_kwargs {dictionary} - Parameters you would pass into Bag of Words constructor as a dictionary

        Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned. 
        """
        report_info = technique_reason_repo['feature']['text']['tfidf']

        if self.data_properties.use_full_data:

            self.data = FeatureTFIDF(
                list_of_cols=list_of_cols, data=self.data, params=params)

            if self.report is not None:
                self.report.ReportTechnique(report_info, [])

            return self.data

        else:

            self.data_properties.train_data, self.data_properties.test_data = FeatureTFIDF(list_of_cols=list_of_cols,
                                                                                           train_data=self.data_properties.train_data,
                                                                                           test_data=self.data_properties.test_data,
                                                                                           params=params)

            if self.report is not None:
                self.report.ReportTechnique(report_info, [])

            return self.data_properties.train_data, self.data_properties.test_data

    def bag_of_words(self, list_of_cols=[], params={}):
        """Creates a matrix of how many times a word appears in a document.

        This function exists in `feature-extraction/text.py`

        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
            params {dictionary} - Parameters you would pass into Bag of Words constructor as a dictionary

        Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned. 
        """
        report_info = technique_reason_repo['feature']['text']['bow']

        if self.data_properties.use_full_data:

            self.data = FeatureBagOfWords(
                list_of_cols, data=self.data, params=params)

            if self.report is not None:
                self.report.ReportTechnique(report_info, [])

            return self.data

        else:

            self.data_properties.train_data, self.data_properties.test_data = FeatureBagOfWords(list_of_cols,
                                                                                                train_data=self.data_properties.train_data,
                                                                                                test_data=self.data_properties.test_data,
                                                                                                params=params)

            if self.report is not None:
                self.report.ReportTechnique(report_info, [])

            return self.data_properties.train_data, self.data_properties.test_data

    def nltk_postag(self, list_of_cols=[]):
        """
        Tag documents with their respective "Part of Speech" tag. These tags classify a word as a
        noun, verb, adjective, etc. A full list and their meaning can be found here:
        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

        
        Args:
            list_of_cols (list, optional): A list of specific columns to apply this technique to. Defaults to [].
        
        Returns:
            Dataframe, *Dataframe: Transformed dataframe with new column of the text columns PoS tagged.

            * Returns 2 Dataframes if Train and Test data is provided.
        """

        report_info = technique_reason_repo['feature']['text']['postag']

        if self.data_properties.use_full_data:
            self.data = NLTKFeaturePoSTag(list_of_cols, data=self.data)

            if self.report is not None:
                self.report.ReportTechnique(report_info, [])

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = NLTKFeaturePoSTag(
                list_of_cols, train_data=self.data_properties.train_data, test_data=self.data_properties.test_data)

            if self.report is not None:
                self.report.ReportTechnique(report_info, [])

            return self.data_properties.train_data, self.data_properties.test_data
