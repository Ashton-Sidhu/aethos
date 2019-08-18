import os

import pandas as pd
import pyautoml
import yaml
from pyautoml.base import MethodBase
from pyautoml.feature_engineering.categorical import *
from pyautoml.feature_engineering.numeric import *
from pyautoml.feature_engineering.text import *

pkg_directory = os.path.dirname(pyautoml.__file__)

with open(f"{pkg_directory}/technique_reasons.yml", 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")


class Feature(MethodBase):

    def __init__(self, data=None, train_data=None, test_data=None, data_properties=None, test_split_percentage=0.2, use_full_data=False, target_field="", report_name=None):

        super().__init__(data=data, train_data=train_data, test_data=test_data, test_split_percentange=test_split_percentage,
                    use_full_data=use_full_data, target_field=target_field, report_name=report_name)

        if self.data_properties.report is not None:
            self.report.WriteHeader("Feature Engineering")


    def onehot_encode(self, list_of_cols, onehot_params={"handle_unknown": "ignore"}):
        """
        Creates a matrix of converted categorical columns into binary columns.
        
        This function exists in `feature-extraction/categorical.py`

        Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])

        Keyword Arguments:            
            data {DataFrame} -- Full dataset (default: {None})
            onehot_params {dictionary} - Parameters you would pass into Onehot encoder constructor as a dictionary (default: {"handle_unknown": "ignore"})

        Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned.  

        """
        report_info = technique_reason_repo['feature']['categorical']['onehotencode']

        if self.data_properties.use_full_data:
            self.data_properties.data = FeatureOneHotEncode(list_of_cols, data=self.data_properties.data, params=onehot_params)

            if self.report is not None:
                self.report.ReportTechnique(report_info, list_of_cols)

            return self.data_properties.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = FeatureOneHotEncode(list_of_cols,
                                                                                                  train_data=self.data_properties.train_data,
                                                                                                  test_data=self.data_properties.test_data,
                                                                                                  params=onehot_params)
            if self.report is not None:
                self.report.ReportTechnique(report_info, list_of_cols)

            return self.data_properties.train_data, self.data_properties.test_data


    def tfidf(self, list_of_cols=[], tfidf_params={}):
        """Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.

        This function exists in `feature-extraction/text.py`

        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
            **params {dictionary} - Parameters you would pass into TFIDF constructor as a dictionary

        Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned. 
        """
        report_info = technique_reason_repo['feature']['text']['tfidf']

        if self.data_properties.use_full_data:
            self.data_properties.data = FeatureTFIDF(
                list_of_cols=list_of_cols, params=tfidf_params, data=self.data_properties.data,)

            if self.report is not None:
                self.report.ReportTechnique(report_info, [])

            return self.data_properties.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = FeatureTFIDF(list_of_cols=list_of_cols,
                                                                                           params=tfidf_params,
                                                                                           train_data=self.data_properties.train_data,
                                                                                           test_data=self.data_properties.test_data,
                                                                                           )

            if self.report is not None:
                self.report.ReportTechnique(report_info, [])

            return self.data_properties.train_data, self.data_properties.test_data


    def bag_of_words(self, list_of_cols=[], bow_params={}):
        """Creates a matrix of how many times a word appears in a document.

        This function exists in `feature-extraction/text.py`

        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: []])
            **params {dictionary} - Parameters you would pass into Bag of Words constructor as a dictionary

        Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned. 
        """
        report_info = technique_reason_repo['feature']['text']['bow']

        if self.data_properties.use_full_data:
            self.data_properties.data = FeatureBagOfWords(
                list_of_cols, params=bow_params, data=self.data_properties.data)

            if self.report is not None:
                self.report.ReportTechnique(report_info, [])

            return self.data_properties.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = FeatureBagOfWords(list_of_cols,
                                                                                                params=bow_params,
                                                                                                train_data=self.data_properties.train_data,
                                                                                                test_data=self.data_properties.test_data,
                                                                                                )

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
            self.data_properties.data = NLTKFeaturePoSTag(list_of_cols, data=self.data_properties.data)

            if self.report is not None:
                self.report.ReportTechnique(report_info, [])

            return self.data_properties.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = NLTKFeaturePoSTag(
                list_of_cols, train_data=self.data_properties.train_data, test_data=self.data_properties.test_data)

            if self.report is not None:
                self.report.ReportTechnique(report_info, [])

            return self.data_properties.train_data, self.data_properties.test_data
