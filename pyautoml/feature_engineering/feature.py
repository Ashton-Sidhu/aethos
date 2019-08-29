import copy
import os

import pandas as pd
import yaml

import pyautoml
from pyautoml.base import MethodBase
from pyautoml.feature_engineering.categorical import *
from pyautoml.feature_engineering.numeric import *
from pyautoml.feature_engineering.text import *
from pyautoml.feature_engineering.util import *
from pyautoml.util import _input_columns

pkg_directory = os.path.dirname(pyautoml.__file__)

with open(f"{pkg_directory}/technique_reasons.yml", 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")


class Feature(MethodBase):

    def __init__(self, data=None, train_data=None, test_data=None, data_properties=None, test_split_percentage=0.2, split=True, target_field="", report_name=None):

        if data_properties is None:        
            super().__init__(data=data, train_data=train_data, test_data=test_data, test_split_percentange=test_split_percentage,
                        split=split, target_field=target_field, report_name=report_name)
        else:
            super().__init__(data=data_properties.data, train_data=data_properties.train_data, test_data=data_properties.test_data, test_split_percentange=test_split_percentage,
                        split=data_properties.split, target_field=data_properties.target_field, report_name=data_properties.report_name)
                        
        if self.data_properties.report is not None:
            self.report.write_header("Feature Engineering")


    def onehot_encode(self, *list_args, list_of_cols=[], **onehot_params):
        """
        Creates a matrix of converted categorical columns into binary columns of ones and zeros.

        If a list of columns is provided use the list, otherwise use arguemnts.
    
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        params : optional
            Parameters you would pass into Bag of Words constructor as a dictionary, by default handle_unknown=ignore}
        
        Returns
        -------
        Feature Object:
            Returns a deep copy of the Feature object.

        Examples
        --------
        >>> feature.onehot_encode('col1', 'col2', 'col3')
        """
        report_info = technique_reason_repo['feature']['categorical']['onehotencode']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self.data_properties.split:
            self.data_properties.data = feature_one_hot_encode(list_of_cols=list_of_cols, data=self.data_properties.data, **onehot_params)

            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)

            return Feature(data_properties=copy.deepcopy(self.data_properties))

        else:
            self.data_properties.train_data, self.data_properties.test_data = feature_one_hot_encode(list_of_cols=list_of_cols,
                                                                                                  train_data=self.data_properties.train_data,
                                                                                                  test_data=self.data_properties.test_data,
                                                                                                  **onehot_params)
            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)

            return Feature(data_properties=copy.deepcopy(self.data_properties))


    def tfidf(self, *list_args, list_of_cols=[], keep_col=False, **tfidf_params):
        """
        Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.

        The higher the score the more important a word is to a document, the lower the score (relative to the other scores)
        the less important a word is to a document.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `feature-extraction/text.py`
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        keep_col : bool, optional
            True if you want to keep the column(s) or False if you want to drop the column(s)
        tfidf_params : optional
            Parameters you would pass into TFIDF constructor as a dictionary, by default {}
        
        Returns
        -------
        Feature Object:
            Returns a deep copy of the Feature object.
        """
        
        report_info = technique_reason_repo['feature']['text']['tfidf']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self.data_properties.split:
            self.data_properties.data = feature_tfidf(
                list_of_cols=list_of_cols, keep_col=keep_col, **tfidf_params, data=self.data_properties.data,)

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return Feature(data_properties=copy.deepcopy(self.data_properties))

        else:
            self.data_properties.train_data, self.data_properties.test_data = feature_tfidf(list_of_cols=list_of_cols,
                                                                                            keep_col = keep_col,
                                                                                           **tfidf_params,
                                                                                           train_data=self.data_properties.train_data,
                                                                                           test_data=self.data_properties.test_data,
                                                                                           )

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return Feature(data_properties=copy.deepcopy(self.data_properties))


    def bag_of_words(self, *list_args, list_of_cols=[], keep_col=False, **bow_params):
        """
        Creates a matrix of how many times a word appears in a document.

        The premise is that the more times a word appears the more the word represents that document.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `feature-extraction/text.py`
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        keep_col : bool, optional
            True if you want to keep the column(s) or False if you want to drop the column(s)
        bow_params : dict, optional
            Parameters you would pass into Bag of Words constructor, by default {}
        
        Returns
        -------
        Feature Object:
            Returns a deep copy of the Feature object.
        """

        report_info = technique_reason_repo['feature']['text']['bow']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self.data_properties.split:
            self.data_properties.data = feature_bag_of_words(
                list_of_cols=list_of_cols, keep_col=keep_col, **bow_params, data=self.data_properties.data)

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return Feature(data_properties=copy.deepcopy(self.data_properties))

        else:
            self.data_properties.train_data, self.data_properties.test_data = feature_bag_of_words(list_of_cols=list_of_cols,
                                                                                                    keep_col=keep_col,
                                                                                                    **bow_params,
                                                                                                    train_data=self.data_properties.train_data,
                                                                                                    test_data=self.data_properties.test_data,
                                                                                                    )

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return Feature(data_properties=copy.deepcopy(self.data_properties))


    def nltk_postag(self, *list_args, list_of_cols=[]):
        """
        Tag documents with their respective "Part of Speech" tag. These tags classify a word as a
        noun, verb, adjective, etc. A full list and their meaning can be found here:
        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

        If a list of columns is provided use the list, otherwise use arguemnts.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        keep_col : bool, optional
            True if you want to keep the column(s) or False if you want to drop the column(s)        
        Returns
        -------
        Feature Object:
            Returns a deep copy of the Feature object.
        """
        
        report_info = technique_reason_repo['feature']['text']['postag']

        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self.data_properties.split:
            self.data_properties.data = nltk_feature_postag(list_of_cols=list_of_cols, data=self.data_properties.data)

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return Feature(data_properties=copy.deepcopy(self.data_properties))

        else:
            self.data_properties.train_data, self.data_properties.test_data = nltk_feature_postag(
                list_of_cols=list_of_cols, train_data=self.data_properties.train_data, test_data=self.data_properties.test_data)

            if self.report is not None:
                self.report.report_technique(report_info, [])

            return Feature(data_properties=copy.deepcopy(self.data_properties))


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
        Feature Object:
            Returns a deep copy of the Feature object.

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
        
        if not self.data_properties.split:    
            self.data_properties.data = apply(func, output_col, data=self.data_properties.data)
    
            if self.report is not None:
                self.report.log(f"Applied function to dataset. {description}")
    
            return Feature(data_properties=copy.deepcopy(self.data_properties))
    
        else:
            self.data_properties.train_data, self.data_properties.test_data = apply(func,
                                                                                                                    output_col,
                                                                                                                    train_data=self.data_properties.train_data,
                                                                                                                    test_data=self.data_properties.test_data)
    
            if self.report is not None:
                self.report.log(f"Added feature {output_col}. {description}")
    
            return Feature(data_properties=copy.deepcopy(self.data_properties))
