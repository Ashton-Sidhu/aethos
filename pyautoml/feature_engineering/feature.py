import copy
import os

import pandas as pd
import pyautoml
import yaml
from pyautoml.base import MethodBase
from pyautoml.feature_engineering.categorical import *
from pyautoml.feature_engineering.numeric import *
from pyautoml.feature_engineering.text import *
from pyautoml.feature_engineering.util import *
from pyautoml.util import (_contructor_data_properties, _input_columns,
                           label_encoder)

pkg_directory = os.path.dirname(pyautoml.__file__)

with open("{}/technique_reasons.yml".format(pkg_directory), 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")


class Feature(MethodBase):

    def __init__(self, step=None, data=None, train_data=None, test_data=None, test_split_percentage=0.2, split=True, target_field="", report_name=None):
        
        _data_properties = _contructor_data_properties(step)

        if _data_properties is None:        
            super().__init__(data=data, train_data=train_data, test_data=test_data, test_split_percentage=test_split_percentage,
                        split=split, target_field=target_field, target_mapping=None, report_name=report_name)
        else:
            super().__init__(data=_data_properties.data, train_data=_data_properties.train_data, test_data=_data_properties.test_data, test_split_percentage=test_split_percentage,
                        split=_data_properties.split, target_field=_data_properties.target_field, target_mapping=_data_properties.target_mapping, report_name=_data_properties.report_name)
                        
        if self._data_properties.report is not None:
            self.report.write_header("Feature Engineering")


    def onehot_encode(self, *list_args, list_of_cols=[], keep_col=True, **onehot_params):
        """
        Creates a matrix of converted categorical columns into binary columns of ones and zeros.

        If a list of columns is provided use the list, otherwise use arguemnts.
    
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        keep_col : bool
            A parameter to specify whether to drop the column being transformed, by default
            keep the column, True

        params : optional
            Parameters you would pass into Bag of Words constructor as a dictionary, by default handle_unknown=ignore}
        
        Returns
        -------
        Feature:
            Returns a deep copy of the Feature object.

        Examples
        --------
        >>> feature.onehot_encode('col1', 'col2', 'col3')
        """
        
        report_info = technique_reason_repo['feature']['categorical']['onehotencode']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = feature_one_hot_encode(list_of_cols=list_of_cols, keep_col=keep_col, data=self._data_properties.data, **onehot_params)
        else:
            self._data_properties.train_data, self._data_properties.test_data = feature_one_hot_encode(list_of_cols=list_of_cols,
                                                                                                    keep_col=keep_col,
                                                                                                    train_data=self._data_properties.train_data,
                                                                                                    test_data=self._data_properties.test_data,
                                                                                                    **onehot_params)
        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()


    def tfidf(self, *list_args, list_of_cols=[], keep_col=True, **tfidf_params):
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
        Feature:
            Returns a deep copy of the Feature object.
        """
        
        report_info = technique_reason_repo['feature']['text']['tfidf']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = feature_tfidf(
                list_of_cols=list_of_cols, keep_col=keep_col, **tfidf_params, data=self._data_properties.data)
        else:
            self._data_properties.train_data, self._data_properties.test_data = feature_tfidf(list_of_cols=list_of_cols,
                                                                                            keep_col = keep_col,
                                                                                           **tfidf_params,
                                                                                           train_data=self._data_properties.train_data,
                                                                                           test_data=self._data_properties.test_data,
                                                                                           )

        if self.report is not None:
            self.report.report_technique(report_info, [])

        return self.copy()


    def bag_of_words(self, *list_args, list_of_cols=[], keep_col=True, **bow_params):
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
        Feature:
            Returns a deep copy of the Feature object.
        """

        report_info = technique_reason_repo['feature']['text']['bow']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = feature_bag_of_words(
                list_of_cols=list_of_cols, keep_col=keep_col, **bow_params, data=self._data_properties.data)
        else:
            self._data_properties.train_data, self._data_properties.test_data = feature_bag_of_words(list_of_cols=list_of_cols,
                                                                                                    keep_col=keep_col,
                                                                                                    **bow_params,
                                                                                                    train_data=self._data_properties.train_data,
                                                                                                    test_data=self._data_properties.test_data,
                                                                                                    )

        if self.report is not None:
            self.report.report_technique(report_info, [])

        return self.copy()


    def postag_nltk(self, *list_args, list_of_cols=[], new_col_name='_postagged'):
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

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_postagged`

        Returns
        -------
        Feature:
            Returns a deep copy of the Feature object.
        """
        
        report_info = technique_reason_repo['feature']['text']['postag']

        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = nltk_feature_postag(list_of_cols=list_of_cols, new_col_name=new_col_name, data=self._data_properties.data)
        else:
            self._data_properties.train_data, self._data_properties.test_data = nltk_feature_postag(
                list_of_cols=list_of_cols, new_col_name=new_col_name, train_data=self._data_properties.train_data, test_data=self._data_properties.test_data)

        if self.report is not None:
            self.report.report_technique(report_info, [])

        return self.copy()


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
        Feature:
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
        
        if not self._data_properties.split:    
            self._data_properties.data = apply(func, output_col, data=self._data_properties.data)        
        else:
            self._data_properties.train_data, self._data_properties.test_data = apply(func,
                                                                                    output_col,
                                                                                    train_data=self._data_properties.train_data,
                                                                                    test_data=self._data_properties.test_data)
    
        if self.report is not None:
            self.report.log("Added feature {}. {}".format(output_col, description))

        return self.copy()

    def encode_labels(self, *list_args, list_of_cols=[]):
        """
        Encode categorical values with value between 0 and n_classes-1.

        Running this function will automatically set the corresponding mapping for the target variable mapping number to the original value.

        Note that this will not work if your test data will have labels that your train data does not.        

        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        
        Returns
        -------
        Feature
            Copy of feature object
        """
    
        report_info = technique_reason_repo['preprocess']['categorical']['label_encode']

        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = label_encoder(
                list_of_cols, data=self._data_properties.data)
        else:
            self._data_properties.train_data, self._data_properties.test_data = label_encoder(
                list_of_cols, train_data=self._data_properties.train_data, test_data=self._data_properties.test_data)

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()
