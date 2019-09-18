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
                           _numeric_input_conditions, label_encoder)

pkg_directory = os.path.dirname(pyautoml.__file__)

with open("{}/technique_reasons.yml".format(pkg_directory), 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")

class Preprocess(MethodBase):

    
    def __init__(self, step=None, data=None, train_data=None, test_data=None, test_split_percentage=0.2, split=True, target_field="", report_name=None):

        _data_properties = _contructor_data_properties(step)

        if _data_properties is None:        
            super().__init__(data=data, train_data=train_data, test_data=test_data, test_split_percentage=test_split_percentage,
                        split=split, target_field=target_field, target_mapping=None, report_name=report_name)
        else:
            super().__init__(data=_data_properties.data, train_data=_data_properties.train_data, test_data=_data_properties.test_data, test_split_percentage=test_split_percentage,
                        split=_data_properties.split, target_field=_data_properties.target_field, target_mapping=_data_properties.target_mapping, report_name=_data_properties.report_name)
                        

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
        Preprocess:
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


    def split_sentences(self, *list_args, list_of_cols=[], new_col_name='_sentences'):
        """
        Splits text data into sentences and saves it into another column for analysis.

        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_sentences`

        Returns
        -------
        Preprocess:
            Returns a deep copy of the Preprocess object.
        """

        report_info = technique_reason_repo['preprocess']['text']['split_sentence']

        list_of_cols = _input_columns(list_args, list_of_cols)        
    
        if not self._data_properties.split:    
            self._data_properties.data = split_sentences(list_of_cols, new_col_name=new_col_name, data=self._data_properties.data)
        else:
            self._data_properties.train_data, self._data_properties.test_data = split_sentences(list_of_cols,
                                                                                            new_col_name=new_col_name,
                                                                                            train_data=self._data_properties.train_data, 
                                                                                            test_data=self._data_properties.test_data)
    
        if self.report is not None:
            self.report.report_technique(report_info)

        return self.copy()

    def stem_nltk(self, *list_args, list_of_cols=[], stemmer='porter', new_col_name='_stemmed'):
        """
        Transforms text to their word stem, base or root form. 
        For example:
            dogs --> dog
            churches --> church
            abaci --> abacus
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        stemmer : str, optional
            Type of NLTK stemmer to use, by default porter

            Current stemming implementations:
                - Porter

            For more information please refer to the NLTK stemming api https://www.nltk.org/api/nltk.stem.html

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_stemmed`
        
        Returns
        -------
        Preprocess
            Copy of preprocess object
        """

        report_info = technique_reason_repo['preprocess']['text']['stem']

        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = nltk_stem(
                list_of_cols=list_of_cols, stemmer=stemmer, new_col_name=new_col_name, data=self._data_properties.data)
        else:
            self._data_properties.train_data, self._data_properties.test_data = nltk_stem(
                list_of_cols=list_of_cols, stemmer=stemmer, new_col_name=new_col_name, train_data=self._data_properties.train_data, test_data=self._data_properties.test_data)

        if self.report is not None:
            self.report.report_technique(report_info)

        return self.copy()

    def split_words_nltk(self, *list_args, list_of_cols=[], regexp='', new_col_name="_tokenized"):
        """
        Splits text into its words using nltk punkt tokenizer by default. 
    
        Default is by spaces and punctuation but if a regex expression is provided, it will use that.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        regexp : str, optional
            Regex expression used to define what a word is.

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_tokenized`
        
        Returns
        -------
        Preprocess
            Copy of preprocess object
        """

        report_info = technique_reason_repo['preprocess']['text']['split_words']

        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = nltk_word_tokenizer(
                list_of_cols=list_of_cols, regexp=regexp, new_col_name=new_col_name, data=self._data_properties.data)
        else:
            self._data_properties.train_data, self._data_properties.test_data = nltk_word_tokenizer(
                list_of_cols=list_of_cols, regexp=regexp, new_col_name=new_col_name, train_data=self._data_properties.train_data, test_data=self._data_properties.test_data)

        if self.report is not None:
            self.report.report_technique(report_info)

        return self.copy()

    def remove_stopwords_nltk(self, *list_args, list_of_cols=[], custom_stopwords=[], new_col_name="_rem_stop"):
        """
        Removes stopwords following the nltk English stopwords list.

        A list of custom words can be provided as well, usually for domain specific words.

        Stop words are generally the most common words in a language
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        custom_stop_words : list, optional
            Custom list of words to also drop with the stop words, must be LOWERCASE, by default []

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_rem_stop`
        
        Returns
        -------
        Preprocess
            Copy of preprocess object
        """

        report_info = technique_reason_repo['preprocess']['text']['remove_stopwords']

        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = nltk_remove_stopwords(
                list_of_cols=list_of_cols, custom_stopwords=custom_stopwords, new_col_name=new_col_name, data=self._data_properties.data)
        else:
            self._data_properties.train_data, self._data_properties.test_data = nltk_remove_stopwords(
                list_of_cols=list_of_cols, custom_stopwords=custom_stopwords, new_col_name=new_col_name, train_data=self._data_properties.train_data, test_data=self._data_properties.test_data)

        if self.report is not None:
            self.report.report_technique(report_info)

        return self.copy()

    def remove_punctuation(self, *list_args, list_of_cols=[], regexp='', exceptions=[], new_col_name='_rem_punct'):
        """
        Removes punctuation from every string entry.

        Defaults to removing all punctuation, but if regex of punctuation is provided, it will remove them.
        
        An example regex would be:

        (\w+\.|\w+)[^,] - Include all words and words with periods after them but don't include commas.
        (\w+\.)|(\w+), would also achieve the same result

        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        regexp : str, optional
            Regex expression used to define what to include.
            
        exceptions : list, optional
            List of punctuation to include in the text, by default []

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_rem_punct`
        
        Returns
        -------
        Preprocess
            Copy of preprocess object
        """

        report_info = technique_reason_repo['preprocess']['text']['remove_punctuation']

        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = remove_punctuation(
                list_of_cols=list_of_cols, regexp=regexp, exceptions=exceptions, new_col_name=new_col_name, data=self._data_properties.data)
        else:
            self._data_properties.train_data, self._data_properties.test_data = remove_punctuation(
                list_of_cols=list_of_cols, regexp=regexp, exceptions=exceptions, new_col_name=new_col_name, train_data=self._data_properties.train_data, test_data=self._data_properties.test_data)

        if self.report is not None:
            self.report.report_technique(report_info)

        return self.copy()

    def encode_labels(self, *list_args, list_of_cols=[]):
        """
        Encode labels with value between 0 and n_classes-1.

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
        Preprocess
            Copy of preprocess object
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
