
import os
import warnings

import pyautoml
import yaml
from pyautoml.base import MethodBase
from pyautoml.modelling.model_types import *
from pyautoml.modelling.text import *
from pyautoml.modelling.util import add_to_queue
from pyautoml.util import (_contructor_data_properties, _input_columns,
                           _validate_model_name)
############################################################
#################### IMPORT MODELS #########################
############################################################
from sklearn.cluster import DBSCAN, KMeans

############################################################
#################### END IMPORT ###########################
############################################################

pkg_directory = os.path.dirname(pyautoml.__file__)

with open("{}/technique_reasons.yml".format(pkg_directory), 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")

## TODO: Implement lazy evaluation of training models.

class Model(MethodBase):

    def __init__(self, step=None, data=None, train_data=None, test_data=None, test_split_percentage=0.2, split=True, target_field="", report_name=None):
        
        _data_properties = _contructor_data_properties(step)

        if _data_properties is None:        
            super().__init__(data=data, train_data=train_data, test_data=test_data, test_split_percentage=test_split_percentage,
                        split=split, target_field=target_field, report_name=report_name)
        else:
            super().__init__(data=_data_properties.data, train_data=_data_properties.train_data, test_data=_data_properties.test_data, test_split_percentage=test_split_percentage,
                        split=_data_properties.split, target_field=_data_properties.target_field, report_name=_data_properties.report_name)
                        
        if self._data_properties.report is not None:
            self.report.write_header("Modelling")

        if target_field:
            if split:
                self._train_target_data = self._data_properties.train_data[self._data_properties.target_field]
                self._test_target_data = self._data_properties.test_data[self._data_properties.test_field]
                self._data_properties.train_data = self._data_properties.train_data.drop([self._data_properties.target_field], axis=1)
                self._data_properties.test_data = self._data_properties.test_data.drop([self._data_properties.target_field], axis=1)
            else:
                self._target_data = self._data_properties.data[self._data_properties.target_field]
                self._data_properties.data = self._data_properties.data.drop([self._data_properties.target_field], axis=1)

        self._models = {}
        self._queued_models = {}

    def __getitem__(self, key):

        if key in self._models:
            return self._models[key]

        return super().__getitem__(key)

    ## Identical copies are made to avoid infinite recursion loop .. better safe than sorry
    def __getattr__(self, key):

        if key in self._models:
            return self._models[key]

        return super().__getattr__(key)

    @property
    def target_data(self):
        """
        Property function for the target data.
        """
        
        if self._data_properties.data is None:
            raise AttributeError("There seems to be nothing here. Try .train_data or .test_data")
        
        return self._target_data

    @target_data.setter
    def target_data(self, value):
        """
        Setter function for the target data.
        """

        self._target_data = value


    @property
    def train_target_data(self):
        """
        Property function for the training target data.
        """
        
        if self._data_properties.train_data is None:
            raise AttributeError("There seems to be nothing here. Try .data")

        return self._train_target_data

    @train_target_data.setter
    def train_target_data(self, value):
        """
        Setter function for the training target data.
        """

        self._train_target_data = value
        
    @property
    def test_target_data(self):
        """
        Property function for the test target data.
        """
        if self._data_properties.train_data is None:
            raise AttributeError("There seems to be nothing here. Try .data")

        return self._test_target_data

    @test_target_data.setter
    def test_target_data(self, value):
        """
        Setter for the test target data.
        """

        self._test_target_data = value

    def train_models(self):
        """ TODO: Implement multi processing running of models """
        return

    @add_to_queue
    def summarize_gensim(self, *list_args, list_of_cols=[], new_col_name="_summarized", model_name="model_summarize_gensim", run=True, **summarizer_kwargs):
        """
        Summarize bodies of text using Gensim's Text Rank algorith. Note that it uses a Text Rank variant as stated here:
        https://radimrehurek.com/gensim/summarization/summariser.html

        The output summary will consist of the most representative sentences and will be returned as a string, divided by newlines.
        
        Parameters
        ----------
        list_of_cols : list, optional
            Column name(s) of text data that you want to summarize
        new_col_name : str, optional
            New column name to be created when applying this technique, by default `_extracted_keywords`
        model_name : str, optional
            Name for this model, default to `model_summarize_gensim`
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        ratio : float, optional
            Number between 0 and 1 that determines the proportion of the number of sentences of the original text to be chosen for the summary.
        word_count : int or None, optional
            Determines how many words will the output contain. If both parameters are provided, the ratio will be ignored.
        split : bool, optional
            If True, list of sentences will be returned. Otherwise joined strings will be returned.

        Returns
        -------
        TextModel
            Resulting model
        """

        if not _validate_model_name(self, model_name):
            raise AttributeError("Invalid model name. Please choose another one.")
    
        report_info = technique_reason_repo['model']['text']['textrank_summarizer']

        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:

            self._data_properties.data = gensim_textrank_summarizer(
                list_of_cols=list_of_cols, new_col_name=new_col_name, data=self._data_properties.data, **summarizer_kwargs)

            if self.report is not None:
                self.report.ReportTechnique(report_info)
            
        else:
            self._data_properties.train_data, self._data_properties.test_data = gensim_textrank_summarizer(
                list_of_cols=list_of_cols, new_col_name=new_col_name, train_data=self._data_properties.train_data, test_data=self._data_properties.test_data, **summarizer_kwargs)

            if self.report is not None:
                self.report.ReportTechnique(report_info)

        self._models[model_name] = TextModel(self)

        return self._models[model_name]        

    @add_to_queue
    def extract_keywords_gensim(self, *list_args, list_of_cols=[], new_col_name="_extracted_keywords", model_name="model_extracted_keywords_gensim", run=True, **keyword_kwargs):
        """
        Extracts keywords using Gensim's implementation of the Text Rank algorithm. 

        Get most ranked words of provided text and/or its combinations.
        
        Parameters
        ----------
        list_of_cols : list, optional
            Column name(s) of text data that you want to summarize
        new_col_name : str, optional
            New column name to be created when applying this technique, by default `_extracted_keywords`
        model_name : str, optional
            Name for this model, default to `model_extract_keywords_gensim`
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        ratio : float, optional
            Number between 0 and 1 that determines the proportion of the number of sentences of the original text to be chosen for the summary.
        words : int, optional
            Number of returned words.        
        split : bool, optional
            If True, list of sentences will be returned. Otherwise joined strings will be returned.
        scores : bool, optional
            Whether score of keyword.
        pos_filter : tuple, optional
            Part of speech filters.
        lemmatize : bool, optional 
            If True - lemmatize words.
        deacc : bool, optional
            If True - remove accentuation.
        
        Returns
        -------
        TextModel
            Resulting model
        """

        if not _validate_model_name(self, model_name):
            raise AttributeError("Invalid model name. Please choose another one.")

        report_info = technique_reason_repo['model']['text']['textrank_keywords']
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:

            self._data_properties.data = gensim_textrank_keywords(
                list_of_cols=list_of_cols, new_col_name=new_col_name, data=self._data_properties.data, **keyword_kwargs)

            if self.report is not None:
                self.report.ReportTechnique(report_info)

        else:
            self._data_properties.train_data, self._data_properties.test_data = gensim_textrank_keywords(
                list_of_cols=list_of_cols, new_col_name=new_col_name, train_data=self._data_properties.train_data, test_data=self._data_properties.test_data, **keyword_kwargs)

            if self.report is not None:
                self.report.ReportTechnique(report_info)

        self._models[model_name] = TextModel(self)

        return self._models[model_name]

    @add_to_queue
    def kmeans(self, model_name="kmeans", new_col_name="kmeans_clusters", run=True, **kmeans_kwargs):
        """
        [summary]
        
        Parameters
        ----------
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        model_name : str, optional
            Name for this model, by default "kmeans"
        new_col_name : str, optional
            Name of column for labels that are generated, by default "kmeans_clusters"
        
        Returns
        -------
        ClusterModel
            ClusterModel object to view results and further analysis
        """

        kmeans = KMeans(**kmeans_kwargs)

        report_info = technique_reason_repo['model']['unsupervised']['kmeans']

        if not self._data_properties.split:
            kmeans.fit(self._data_properties.data)

            self._data_properties.data[new_col_name] = kmeans.labels_

        else:
            kmeans.fit(self._data_properties.train_data)

            self._data_properties.train_data[new_col_name] = kmeans.labels_
            self._data_properties.test_data[new_col_name] = kmeans.predict(
                self._data_properties.test_data)

        self._models[model_name] = ClusterModel(self)

        return self._models[model_name]

    @add_to_queue
    def dbscan(self, model_name="dbscan", new_col_name="dbscan_clusters", run=True, **dbscan_kwargs):
        """
        [summary]
        
        Parameters
        ----------
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        model_name : str, optional
            Name for this model, by default "dbscan"
        new_col_name : str, optional
            Name of column for labels that are generated, by default "dbscan_clusters"
        
        Returns
        -------
        ClusterModel
            ClusterModel object to view results and further analysis
        """

        dbscan = DBSCAN(**dbscan_kwargs)

        report_info = technique_reason_repo['model']['unsupervised']['kmeans']

        if not self._data_properties.split:
            dbscan.fit(self._data_properties.data)
            self._data_properties.data[new_col_name] = dbscan.labels_

        else:
            warnings.warn(
                'DBSCAN has no predict method, so training and testing data was combined and DBSCAN was trained on the full data. To access results you can use `.data`.')

            full_data = self._data_properties.train_data.append(
                self._data_properties.test_data, ignore_index=True)
            dbscan.fit(full_data)
            full_data[new_col_name] = dbscan.labels_
            self._data_properties.data = full_data

        self._models[model_name] = ClusterModel(self)

        return self._models[model_name]
