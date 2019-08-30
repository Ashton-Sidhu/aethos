import copy
import os

import pandas as pd
import pyautoml
import yaml
from pyautoml.base import MethodBase
from pyautoml.cleaning.categorical import *
from pyautoml.cleaning.numeric import *
from pyautoml.cleaning.util import *
from pyautoml.util import _contructor_data_properties, _input_columns

pkg_directory = os.path.dirname(pyautoml.__file__)

with open(f"{pkg_directory}/technique_reasons.yml", 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")

class Clean(MethodBase):

    
    def __init__(self, step=None, data=None, train_data=None, test_data=None, _data_properties=None, test_split_percentage=0.2, split=True, target_field="", report_name=None):   
        
        _data_properties = _contructor_data_properties(step)

        if _data_properties is None:        
            super().__init__(data=data, train_data=train_data, test_data=test_data, test_split_percentage=test_split_percentage,
                        split=split, target_field=target_field, report_name=report_name)
        else:
            super().__init__(data=_data_properties.data, train_data=_data_properties.train_data, test_data=_data_properties.test_data, test_split_percentage=test_split_percentage,
                        split=_data_properties.split, target_field=_data_properties.target_field, report_name=_data_properties.report_name)
        
        if self._data_properties.report is not None:
            self.report.write_header("Cleaning")


    def remove_columns(self, threshold: float):
        """
        Remove columns from the dataframe that have more than the threshold value of missing columns.
        Example: Remove columns where > 50% of the data is missing.

        This function exists in `clean/utils.py`
        
        Parameters
        ----------
        threshold : float
            Value between 0 and 1 that describes what percentage of a column can be missing values.
        
        Returns
        -------
        Clean Object:
            Returns a deep copy of the Clean object.
        """

        report_info = technique_reason_repo['clean']['general']['remove_columns']

        if not self._data_properties.split:            
            #Gather original data information
            original_columns = set(list(self._data_properties.data.columns))

            self._data_properties.data = remove_columns_threshold(threshold, data=self._data_properties.data)

            #Write to report
            if self.report is not None:
                new_columns = original_columns.difference(self._data_properties.data.columns)
                self.report.report_technique(report_info, new_columns)

            return Clean(copy.deepcopy(self._data_properties))

        else:
            #Gather original data information
            original_columns = set(list(self._data_properties.train_data.columns))

            self._data_properties.train_data, self._data_properties.test_data = remove_columns_threshold(threshold,
                                                                                                        train_data=self._data_properties.train_data,
                                                                                                        test_data=self._data_properties.test_data)

            if self.report is not None:
                new_columns = original_columns.difference(self._data_properties.train_data.columns)
                self.report.report_technique(report_info, new_columns)

            return Clean(copy.deepcopy(self._data_properties))


    def remove_rows(self, threshold: float):
        """
        Remove rows from the dataframe that have more than the threshold value of missing rows.
        Example: Remove rows where > 50% of the data is missing.

        This function exists in `clean/utils.py`.

        Parameters
        ----------
        threshold : float
            Value between 0 and 1 that describes what percentage of a row can be missing values.
        
        Returns
        -------
        Clean Object:
            Returns a deep copy of the Clean object.
        """

        report_info = technique_reason_repo['clean']['general']['remove_rows']

        if not self._data_properties.split:
            self._data_properties.data = remove_rows_threshold(threshold, data=self._data_properties.data)

            #Write to report
            if self.report is not None:            
                self.report.report_technique(report_info, [])

            return Clean(copy.deepcopy(self._data_properties))

        else:
            self._data_properties.train_data, self._data_properties.test_data = remove_rows_threshold(threshold,
                                                                                                    train_data=self._data_properties.train_data,
                                                                                                    test_data=self._data_properties.test_data)

            #Write to report
            if self.report is not None:            
                self.report.report_technique(report_info, [])                                                                                    

            return Clean(copy.deepcopy(self._data_properties))
    
    def replace_missing_mean(self, *list_args, list_of_cols=[]):
        """
        Replaces missing values in every numeric column with the mean of that column.

        If no columns are supplied, missing values will be replaced with the mean in every numeric column.

        Mean: Average value of the column. Effected by outliers.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/numeric.py` as `replace_missing_mean_median_mode`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to
        list_of_cols : list, optional
            Specific columns to apply this technique to, by default []
        
        Returns
        -------
        Clean Object:
            Returns a deep copy of the Clean object.
        """

        report_info = technique_reason_repo['clean']['numeric']['mean']
        
        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = replace_missing_mean_median_mode(list_of_cols=list_of_cols, strategy="mean", data=self._data_properties.data)

            #Write to report
            if self.report is not None:            
                if list_of_cols:
                    self.report.report_technique(report_info, list_of_cols)
                else:
                    list_of_cols = _numeric_input_conditions(list_of_cols, self._data_properties.data, None)
                    self.report.report_technique(report_info, list_of_cols)

            return Clean(copy.deepcopy(self._data_properties))

        else:
            self._data_properties.train_data, self._data_properties.test_data = replace_missing_mean_median_mode(list_of_cols=list_of_cols,
                                                                                                            strategy="mean",                                                                                                            
                                                                                                            train_data=self._data_properties.train_data,
                                                                                                            test_data=self._data_properties.test_data)
            
            if self.report is not None:
                if list_of_cols:
                    self.report.report_technique(report_info, list_of_cols)
                else:
                    list_of_cols = _numeric_input_conditions(list_of_cols, None, self._data_properties.train_data)
                    self.report.report_technique(report_info, list_of_cols)

            return Clean(copy.deepcopy(self._data_properties))

    def replace_missing_median(self, *list_args, list_of_cols=[]):
        """
        Replaces missing values in every numeric column with the median of that column.

        If no columns are supplied, missing values will be replaced with the mean in every numeric column.

        Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/numeric.py` as `replace_missing_mean_median_mode`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            Specific columns to apply this technique to., by default []
        
        Returns
        -------
        Clean Object:
            Returns a deep copy of the Clean object.
        """

        report_info = technique_reason_repo['clean']['numeric']['median']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = replace_missing_mean_median_mode(list_of_cols=list_of_cols, strategy="median", data=self._data_properties.data)
            
            if self.report is not None:
                if list_of_cols:
                    self.report.report_technique(report_info, list_of_cols)
                else:
                    list_of_cols = _numeric_input_conditions(list_of_cols, self._data_properties.data, None)
                    self.report.report_technique(report_info, list_of_cols)

            return Clean(copy.deepcopy(self._data_properties))

        else:
            self._data_properties.train_data, self._data_properties.test_data = replace_missing_mean_median_mode(list_of_cols=list_of_cols,
                                                                                                            strategy="median",                                                                                                            
                                                                                                            train_data=self._data_properties.train_data,
                                                                                                            test_data=self._data_properties.test_data)

            if self.report is not None:
                if list_of_cols:
                    self.report.report_technique(report_info, list_of_cols)
                else:
                    list_of_cols = _numeric_input_conditions(list_of_cols, None, self._data_properties.train_data)
                    self.report.report_technique(report_info, list_of_cols)

            return Clean(copy.deepcopy(self._data_properties))

    def replace_missing_mostcommon(self, *list_args, list_of_cols=[]):
        """
        Replaces missing values in every numeric column with the most common value of that column

        Mode: Most common value.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/numeric.py` as `replace_missing_mean_median_mode`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        
        Returns
        -------
        Clean Object:
            Returns a deep copy of the Clean object.
        """
       
        report_info = technique_reason_repo['clean']['numeric']['mode']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = replace_missing_mean_median_mode(list_of_cols=list_of_cols, strategy="most_frequent", data=self._data_properties.data)

            if self.report is not None:
                if list_of_cols:
                    self.report.report_technique(report_info, list_of_cols)
                else:
                    list_of_cols = _numeric_input_conditions(list_of_cols, self._data_properties.data, None)
                    self.report.report_technique(report_info, list_of_cols)

            return Clean(copy.deepcopy(self._data_properties))

        else:
            self._data_properties.train_data, self._data_properties.test_data = replace_missing_mean_median_mode(list_of_cols=list_of_cols,
                                                                                                            strategy="most_frequent",                                                                                                            
                                                                                                            train_data=self._data_properties.train_data,
                                                                                                            test_data=self._data_properties.test_data)
            if self.report is not None:
                if list_of_cols:
                    self.report.report_technique(report_info, list_of_cols)
                else:
                    list_of_cols = _numeric_input_conditions(list_of_cols, None, self._data_properties.train_data)
                    self.report.report_technique(report_info, list_of_cols)

            return Clean(copy.deepcopy(self._data_properties))

    def replace_missing_constant(self, *list_args, list_of_cols=[], constant=0, col_mapping=None):
        """
        Replaces missing values in every numeric column with a constant.

        If no columns are supplied, missing values will be replaced with the mean in every numeric column.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/numeric.py` as `replace_missing_constant`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        constant : int or float, optional
            Numeric value to replace all missing values with , by default 0
        col_mapping : dict, optional
            Dictionary mapping {'ColumnName': `constant`}, by default None
        
        Returns
        -------
        Clean Object:
            Returns a deep copy of the Clean object.

        Examples
        --------
        >>> replace_missing_constant(col_mapping={'a': 1, 'b': 2, 'c': 3})

        >>> replace_missing_constant('col1', 'col2', constant=2)
        """

        report_info = technique_reason_repo['clean']['numeric']['constant']

        if col_mapping:
            col_to_constant = col_mapping
        else:
            ## If a list of columns is provided use the list, otherwise use arguemnts.
            col_to_constant = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = replace_missing_constant(col_to_constant=col_to_constant, constant=constant, data=self._data_properties.data)

            if self.report is not None:
                if not col_to_constant:
                    self.report.report_technique(report_info, self._data_properties.data.columns)
                else:
                    self.report.report_technique(report_info, list(col_to_constant))

            return Clean(copy.deepcopy(self._data_properties))

        else:
            self._data_properties.train_data, self._data_properties.test_data = replace_missing_constant(col_to_constant=col_to_constant,
                                                                                                    constant=constant,                                                                                                    
                                                                                                    train_data=self._data_properties.train_data,
                                                                                                    test_data=self._data_properties.test_data)

            if self.report is not None:
                if not col_to_constant:
                    self.report.report_technique(report_info, self._data_properties.train_data.columns)
                else:
                    self.report.report_technique(report_info, list(col_to_constant))

            return Clean(copy.deepcopy(self._data_properties))


    def replace_missing_new_category(self, *list_args, list_of_cols=[], new_category=None, col_mapping=None):
        """
        Replaces missing values in categorical column with its own category. The categories can be autochosen
        from the defaults set.

        For numeric categorical columns default values are: -1, -999, -9999
        For string categorical columns default values are: "Other", "Unknown", "MissingDataCategory"

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/categorical.py` as `replace_missing_new_category`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        new_category : str, int, or float, optional
            Category to replace missing values with, by default None
        col_mapping : dict, optional
           Dictionary mapping {'ColumnName': `constant`}, by default None
        
        Returns
        -------
        Clean Object:
            Returns a deep copy of the Clean object.

        Examples
        --------
        >>> replace_missing_new_category(col_mapping={'col1': "Green", 'col2': "Canada", 'col3': "December"})

        >>> replace_missing_new_category('col1', 'col2', 'col3', new_category='Blue')
        """

        report_info = technique_reason_repo['clean']['categorical']['new_category']
        
        ## If dictionary mapping is provided, use that otherwise use column
        if col_mapping:
            col_to_category = col_mapping
        else:
            ## If a list of columns is provided use the list, otherwise use arguemnts.
            col_to_category = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = replace_missing_new_category(col_to_category=col_to_category, constant=new_category, data=self._data_properties.data)

            if self.report is not None:
                if not col_to_category:
                    self.report.report_technique(report_info, self._data_properties.data.columns)
                else:
                    self.report.report_technique(report_info, list(col_to_category))

            return Clean(copy.deepcopy(self._data_properties))

        else:
            self._data_properties.train_data, self._data_properties.test_data = replace_missing_new_category(col_to_category=col_to_category,
                                                                                                        constant=new_category,                                                                                                                                                                                                            
                                                                                                        train_data=self._data_properties.train_data,
                                                                                                        test_data=self._data_properties.test_data)

            if self.report is not None:
                if col_to_category is None:
                    self.report.report_technique(report_info, self._data_properties.train_data.columns)
                else:
                    self.report.report_technique(report_info, list(col_to_category))                                                                                                   

            return Clean(copy.deepcopy(self._data_properties))


    def replace_missing_remove_row(self, *list_args, list_of_cols=[]):
        """
        Remove rows where the value of a column for those rows is missing.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/categorical.py` as `replace_missing_remove_row`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        Returns
        -------
        Clean Object:
            Returns a deep copy of the Clean object.
        """

        report_info = technique_reason_repo['clean']['categorical']['remove_rows']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        if not self._data_properties.split:
            self._data_properties.data = replace_missing_remove_row(list_of_cols, data=self._data_properties.data)

            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)

            return Clean(copy.deepcopy(self._data_properties))

        else:
            self._data_properties.train_data, self._data_properties.test_data = replace_missing_remove_row(list_of_cols,                                                                                                    
                                                                                                    train_data=self._data_properties.train_data,
                                                                                                    test_data=self._data_properties.test_data)                                                                                        

            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)

            return Clean(copy.deepcopy(self._data_properties))


    def remove_duplicate_rows(self, *list_args, list_of_cols=[]):
        """
        Remove rows from the data that are exact duplicates of each other and leave only 1.
        This can be used to reduce processing time or performance for algorithms where
        duplicates have no effect on the outcome (i.e DBSCAN)

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/util.py` as `remove_duplicate_rows`.
       
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
       
        Returns
        -------
        Clean Object:
            Returns a deep copy of the Clean object.
        """
    
        report_info = technique_reason_repo['clean']['general']['remove_duplicate_rows']
        
        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)
   
        if not self._data_properties.split:
            self._data_properties.data = remove_duplicate_rows(list_of_cols=list_of_cols, data=self._data_properties.data)

            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)
    
            return Clean(copy.deepcopy(self._data_properties))
    
        else:
            self._data_properties.train_data, self._data_properties.test_data = remove_duplicate_rows(list_of_cols=list_of_cols,
                                                                                                train_data=self._data_properties.train_data,
                                                                                                test_data=self._data_properties.test_data)

            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)

            return Clean(copy.deepcopy(self._data_properties))


    def remove_duplicate_columns(self):
        """
        Remove columns from the data that are exact duplicates of each other and leave only 1.
        
        Returns
        -------
        Clean Object:
            Returns a deep copy of the Clean object.
        """
    
        report_info = technique_reason_repo['clean']['general']['remove_duplicate_columns']
    
        if not self._data_properties.split:
            self._data_properties.data = remove_duplicate_columns(data=self._data_properties.data)

            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)
    
            return Clean(copy.deepcopy(self._data_properties))
    
        else:
            self._data_properties.train_data, self._data_properties.test_data = remove_duplicate_columns(train_data=self._data_properties.train_data,
                                                                                                        test_data=self._data_properties.test_data)

            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)

            return Clean(copy.deepcopy(self._data_properties))


    def replace_missing_random_discrete(self, *list_args, list_of_cols=[]):
        """
        Replace missing values in with a random number based off the distribution (number of occurences) 
        of the data.

        If a list of columns is provided use the list, otherwise use arguemnts.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        
        Returns
        -------
        Clean Object:
            Returns a deep copy of the Clean object.

        Examples
        --------
        >>> For example if your data was [5, 5, NaN, 1, 2]
        >>> There would be a 50% chance that the NaN would be replaced with a 5, a 25% chance for 1 and a 25% chance for 2.

        """
    
        report_info = technique_reason_repo['clean']['general']['random_discrete']
        
        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)
        
        if not self._data_properties.split:   
            self._data_properties.data = replace_missing_random_discrete(list_of_cols, data=self._data_properties.data)

            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)
    
            return Clean(copy.deepcopy(self._data_properties))
    
        else:
            self._data_properties.train_data, self._data_properties.test_data = replace_missing_random_discrete(list_of_cols,
                                                                                                            train_data=self._data_properties.train_data,
                                                                                                            test_data=self._data_properties.test_data)
    
            if self.report is not None:
                self.report.report_technique(report_info, list_of_cols)
    
            return Clean(copy.deepcopy(self._data_properties))
