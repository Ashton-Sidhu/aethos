import os

import pandas as pd
import yaml

import pyautoml
from pyautoml.base import MethodBase
from pyautoml.feature_engineering.categorical import *
from pyautoml.feature_engineering.numeric import *
from pyautoml.feature_engineering.text import *
from pyautoml.feature_engineering.util import *

pkg_directory = os.path.dirname(pyautoml.__file__)

with open(f"{pkg_directory}/technique_reasons.yml", 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")


class Feature(MethodBase):

    def __init__(self, data=None, train_data=None, test_data=None, data_properties=None, test_split_percentage=0.2, use_full_data=False, target_field="", report_name=None):

        if data_properties is None:        
            super().__init__(data=data, train_data=train_data, test_data=test_data, test_split_percentange=test_split_percentage,
                        use_full_data=use_full_data, target_field=target_field, report_name=report_name)
        else:
            super().__init__(data=data_properties.data, train_data=data_properties.train_data, test_data=data_properties.test_data, test_split_percentange=test_split_percentage,
                        use_full_data=data_properties.use_full_data, target_field=data_properties.target_field, report_name=data_properties.report)
                        
        if self.data_properties.report is not None:
            self.report.write_header("Feature Engineering")


    def onehot_encode(self, list_of_cols: list, onehot_params={"handle_unknown": "ignore"}):
        """
        Creates a matrix of converted categorical columns into binary columns of ones and zeros.
    
        Parameters
        ----------
        list_of_cols : list
            A list of specific columns to apply this technique to.
        params : dict, optional
            Parameters you would pass into Bag of Words constructor as a dictionary, by default {"handle_unknown": "ignore"}
        
        Returns
        -------
        Dataframe, *Dataframe
            Transformed dataframe with rows with a missing values in a specific column are missing

        * Returns 2 Dataframes if Train and Test data is provided. 
        """
        report_info = technique_reason_repo['feature']['categorical']['onehotencode']

        if self.data_properties.use_full_data:
            self.data_properties.data = feature_one_hot_encode(list_of_cols, data=self.data_properties.data, params=onehot_params)

            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)

            return self.data_properties.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = feature_one_hot_encode(list_of_cols,
                                                                                                  train_data=self.data_properties.train_data,
                                                                                                  test_data=self.data_properties.test_data,
                                                                                                  params=onehot_params)
            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)

            return self.data_properties.train_data, self.data_properties.test_data


    def tfidf(self, list_of_cols=[], tfidf_params={}):
        """
        Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.

        The higher the score the more important a word is to a document, the lower the score (relative to the other scores)
        the less important a word is to a document.

        This function exists in `feature-extraction/text.py`
        
        Parameters
        ----------
        list_of_cols : list, optional
            A list of specific columns to apply this technique to, by default []
        tfidf_params : dict, optional
            Parameters you would pass into TFIDF constructor as a dictionary, by default {}
        
        Returns
        -------
        Dataframe, *Dataframe
            Transformed dataframe with rows with a missing values in a specific column are missing

        * Returns 2 Dataframes if Train and Test data is provided. 
        """
        
        report_info = technique_reason_repo['feature']['text']['tfidf']

        if self.data_properties.use_full_data:
            self.data_properties.data = feature_tfidf(
                list_of_cols=list_of_cols, params=tfidf_params, data=self.data_properties.data,)

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return self.data_properties.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = feature_tfidf(list_of_cols=list_of_cols,
                                                                                           params=tfidf_params,
                                                                                           train_data=self.data_properties.train_data,
                                                                                           test_data=self.data_properties.test_data,
                                                                                           )

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return self.data_properties.train_data, self.data_properties.test_data


    def bag_of_words(self, list_of_cols=[], bow_params={}):
        """
        Creates a matrix of how many times a word appears in a document.

        The premise is that the more times a word appears the more the word represents that document.

        This function exists in `feature-extraction/text.py`
        
        Parameters
        ----------
        list_of_cols : list, optional
            A list of specific columns to apply this technique to, by default []
        bow_params : dict, optional
            Parameters you would pass into Bag of Words constructor as a dictionary, by default {}
        
        Returns
        -------
        Dataframe, *Dataframe
            Transformed dataframe with rows with a missing values in a specific column are missing

        * Returns 2 Dataframes if Train and Test data is provided. 
        """

        report_info = technique_reason_repo['feature']['text']['bow']

        if self.data_properties.use_full_data:
            self.data_properties.data = feature_bag_of_words(
                list_of_cols, params=bow_params, data=self.data_properties.data)

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return self.data_properties.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = feature_bag_of_words(list_of_cols,
                                                                                                params=bow_params,
                                                                                                train_data=self.data_properties.train_data,
                                                                                                test_data=self.data_properties.test_data,
                                                                                                )

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return self.data_properties.train_data, self.data_properties.test_data


    def nltk_postag(self, list_of_cols=[]):
        """
        Tag documents with their respective "Part of Speech" tag. These tags classify a word as a
        noun, verb, adjective, etc. A full list and their meaning can be found here:
        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        
        Parameters
        ----------
        list_of_cols : list, optional
            A list of specific columns to apply this technique to, by default []
        
        Returns
        -------
        Dataframe, *Dataframe
            Transformed dataframe with rows with a missing values in a specific column are missing

        * Returns 2 Dataframes if Train and Test data is provided. 
        """
        
        report_info = technique_reason_repo['feature']['text']['postag']

        if self.data_properties.use_full_data:
            self.data_properties.data = nltk_feature_postag(list_of_cols, data=self.data_properties.data)

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return self.data_properties.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = nltk_feature_postag(
                list_of_cols, train_data=self.data_properties.train_data, test_data=self.data_properties.test_data)

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return self.data_properties.train_data, self.data_properties.test_data


    def apply(self, func, output_col: str, description=''):
        """
        Calls pandas apply function. Will apply the function to your dataset, or
        both your training and testing dataset.
        
        Parameters
        ----------
        func : Function pointer
            Function describing the transformation for the new column
        output_col : str
            New column name
        description : str, optional
            Description of the new column to be logged into the report, by default ''
        
        Returns
        -------
        Dataframe, *Dataframe
            Transformed dataframe with rows with a missing values in a specific column are missing

        * Returns 2 Dataframes if Train and Test data is provided. 

        Examples
        --------
        >>>     col1  col2  col3 
            0     1     0     1       
            1     0     2     0       
            2     1     0     1
        >>> feature.apply(lambda x: x['col1'] > 0, 'col4')
        >>>     col1  col2  col3  col4 
            0     1     0     1     1       
            1     0     2     0     0  
            2     1     0     1     1
        """
        
        if self.data_properties.use_full_data:
    
            self.data_properties.data.loc[:, output_col] = apply(func, output_col, data=self.data_properties.data)
    
            if self.report is not None:
                self.report.log(f"Applied function to dataset. {description}")
    
            return self.data_properties.data
    
        else:
            self.data_properties.train_data.loc[:, output_col], self.data_properties.test_data.loc[:, output_col] = apply(func,
                                                                                                                    output_col,
                                                                                                                    train_data=self.data_properties.train_data,
                                                                                                                    test_data=self.data_properties.test_data)
    
            if self.report is not None:
                self.report.log(f"Applied function to train and test dataset. {description}")
    
            return self.data_properties.train_data, self.data_properties.test_data
