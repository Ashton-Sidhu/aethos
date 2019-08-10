import os

import pandas as pd
import pyautoml
import yaml
from pyautoml.cleaning.categorical import *
from pyautoml.cleaning.numeric import *
from pyautoml.cleaning.util import *
from pyautoml.data.data import Data
from pyautoml.util import (CheckMissingData, GetListOfCols, SplitData,
                           _FunctionInputValidation,
                           _NumericFunctionInputConditions)

pkg_directory = os.path.dirname(pyautoml.__file__)

with open(f"{pkg_directory}/technique_reasons.yml", 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")

class Clean():

    
    def __init__(self, data=None, train_data=None, test_data=None, test_split_percentage=0.2, use_full_data=False, target_field="", report_name=None):        

        if not _FunctionInputValidation(data, train_data, test_data):
            raise ValueError("Error initialzing constructor, please provide one of either data or train_data and test_data, not both.")

        self.data_properties = Data(data, train_data, test_data, use_full_data=use_full_data, target_field=target_field, report_name=report_name)

        if data is not None:
            self.data = data
            self.data_properties.train_data, self.data_properties.test_data = SplitData(self.data, test_split_percentage)
        else:
            # Override user input for safety.
            self.data_properties.use_full_data = False       

        if self.data_properties.report is None:
            self.report = None
        else:
            self.report = self.data_properties.report
            self.report.WriteHeader("Cleaning")

        self.train_data = self.data_properties.train_data
        self.test_data = self.data_properties.test_data

    def __repr__(self):
        if self.data_properties.use_full_data:
            return self.data.__repr__()
        else:
            return self.data_properties.train_data.__repr__()

    @property
    def missing_values(self):       
        """
        Property that displays every column and how many missing values it has along with percentage.
        """
        if self.data_properties.use_full_data:
            missing_values = MissingData(self.data)
        else:
            missing_values = MissingData(self.data_properties.train_data, self.data_properties.test_data)
        
        for item in missing_values:
            print(item.__repr__())    


    def remove_columns(self, threshold):
        """Remove columns from the dataframe that have more than the threshold value of missing columns.
        Example: Remove columns where > 50% of the data is missing.

        This function exists in `clean/utils.py`.
        
        Arguments:
            threshold {[float]} -- Value between 0 and 1 that describes what percentage of a column can be missing values.

        Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned. 
        """

        report_info = technique_reason_repo['clean']['general']['remove_columns']

        if self.data_properties.use_full_data:            
            #Gather original data information
            original_columns = set(list(self.data.columns))

            self.data = RemoveColumns(threshold, data=self.data)

            #Write to report
            if self.report is not None:
                new_columns = original_columns.difference(self.data.columns)
                self.report.ReportTechnique(report_info, new_columns)

            return self.data

        else:
            #Gather original data information
            original_columns = set(list(self.train_data.columns))

            self.data_properties.train_data, self.data_properties.test_data = RemoveColumns(threshold,
                                                                                            train_data=self.data_properties.train_data,
                                                                                            test_data=self.data_properties.test_data)

            if self.report is not None:
                new_columns = original_columns.difference(self.data_properties.train_data.columns)
                self.report.ReportTechnique(report_info, new_columns)

            return self.data_properties.train_data, self.data_properties.test_data


    def remove_rows(self, threshold):
        """Remove rows from the dataframe that have more than the threshold value of missing rows.
        Example: Remove rows where > 50% of the data is missing.

        This function exists in `clean/utils.py`.
        
        Arguments:
            threshold {[float]} -- Value between 0 and 1 that describes what percentage of a row can be missing values.

        Returns:
            [DataFrame],  DataFrame] -- Dataframe(s) missing values replaced by the method. If train and test are provided then the cleaned version 
            of both are returned. 
        """
        
        report_info = technique_reason_repo['clean']['general']['remove_rows']

        if self.data_properties.use_full_data:
            self.data = RemoveRows(threshold, data=self.data)

            #Write to report
            if self.report is not None:            
                self.report.ReportTechnique(report_info, [])

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = RemoveRows(threshold,
                                                                                        train_data=self.data_properties.train_data,
                                                                                        test_data=self.data_properties.test_data)

            #Write to report
            if self.report is not None:            
                self.report.ReportTechnique(report_info, [])                                                                                    

            return self.data_properties.train_data, self.data_properties.test_data
    
    def replace_missing_mean(self, list_of_cols=[]):
        """Replaces missing values in every numeric column with the mean of that column.

        Mean: Average value of the column. Effected by outliers.

        This function exists in `clean/numeric.py` as `ReplaceMissingMeanMedianMode`.
       
        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the list_of_cols overrides the columns in field_types
                                    Example: if list_of_cols is provided and override is true, the technique will only be applied
                                    to the the columns in list_of_cols (default: {False})
        """

        report_info = technique_reason_repo['clean']['numeric']['mean']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingMeanMedianMode("mean", list_of_cols, data=self.data)

            #Write to report
            if self.report is not None:            
                if list_of_cols:
                    self.report.ReportTechnique(report_info, list_of_cols)
                else:
                    list_of_cols = _NumericFunctionInputConditions(list_of_cols, self.data, None)
                    self.report.ReportTechnique(report_info, list_of_cols)

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = ReplaceMissingMeanMedianMode("mean",
                                                                                                            list_of_cols=list_of_cols,
                                                                                                            train_data=self.data_properties.train_data,
                                                                                                            test_data=self.data_properties.test_data)
            
            if self.report is not None:
                if list_of_cols:
                    self.report.ReportTechnique(report_info, list_of_cols)
                else:
                    list_of_cols = _NumericFunctionInputConditions(list_of_cols, None, self.data_properties.train_data)
                    self.report.ReportTechnique(report_info, list_of_cols)

            return self.data_properties.train_data, self.data_properties.test_data

    def replace_missing_median(self, list_of_cols=[]):
        """Replaces missing values in every numeric column with the median of that column.

        Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.

        This function exists in `clean/numeric.py` as `ReplaceMissingMeanMedianMode`.
        
        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the list_of_cols overrides the columns in field_types
                                    Example: if list_of_cols is provided and override is true, the technique will only be applied
                                    to the the columns in list_of_cols (default: {False})                       
        """

        report_info = technique_reason_repo['clean']['numeric']['median']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingMeanMedianMode("median", list_of_cols, data=self.data)
            
            if self.report is not None:
                if list_of_cols:
                    self.report.ReportTechnique(report_info, list_of_cols)
                else:
                    list_of_cols = _NumericFunctionInputConditions(list_of_cols, self.data, None)
                    self.report.ReportTechnique(report_info, list_of_cols)

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = ReplaceMissingMeanMedianMode("median",
                                                                                                            list_of_cols=list_of_cols,
                                                                                                            train_data=self.data_properties.train_data,
                                                                                                            test_data=self.data_properties.test_data)

            if self.report is not None:
                if list_of_cols:
                    self.report.ReportTechnique(report_info, list_of_cols)
                else:
                    list_of_cols = _NumericFunctionInputConditions(list_of_cols, None, self.data_properties.train_data)
                    self.report.ReportTechnique(report_info, list_of_cols)

            return self.data_properties.train_data, self.data_properties.test_data

    def replace_missing_mode(self, list_of_cols=[]):
        """Replaces missing values in every numeric column with the mode of that column

        Mode: Most common number in a list of numbers.

        This function exists in `clean/numeric.py` as `ReplaceMissingMeanMedianMode`.
        
        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the list_of_cols overrides the columns in field_types
                                    Example: if list_of_cols is provided and override is true, the technique will only be applied
                                    to the the columns in list_of_cols (default: {False})      
        """
       
        report_info = technique_reason_repo['clean']['numeric']['mode']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingMeanMedianMode("most_frequent", list_of_cols, data=self.data)

            if self.report is not None:
                if list_of_cols:
                    self.report.ReportTechnique(report_info, list_of_cols)
                else:
                    list_of_cols = _NumericFunctionInputConditions(list_of_cols, self.data, None)
                    self.report.ReportTechnique(report_info, list_of_cols)

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = ReplaceMissingMeanMedianMode("most_frequent",
                                                                                                            list_of_cols=list_of_cols,
                                                                                                            train_data=self.data_properties.train_data,
                                                                                                            test_data=self.data_properties.test_data)
            if self.report is not None:
                if list_of_cols:
                    self.report.ReportTechnique(report_info, list_of_cols)
                else:
                    list_of_cols = _NumericFunctionInputConditions(list_of_cols, None, self.data_properties.train_data)
                    self.report.ReportTechnique(report_info, list_of_cols)

            return self.data_properties.train_data, self.data_properties.test_data

    def replace_missing_constant(self, constant=0, col_to_constant=None):
        """Replaces missing values in every numeric column with a constant.

        This function exists in `clean/numeric.py` as `ReplaceMissingConstant`.
        
        Keyword Arguments:
            constant {int or float} -- Numeric value to replace all missing values with (default: {0})
            col_to_constant {list} or {dict} -- A list of specific columns to apply this technique to or a dictionary
            mapping {Column Name: `constant`}. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                    Example: if custom_cols is provided and override is true, the technique will only be applied
                                    to the the columns in custom_cols (default: {False})
        
        Examples:

        >>>> replace_missing_constant({'a': 1, 'b': 2, 'c': 3})
        >>>> replace_missing_constant(1, ['a', 'b', 'c'])
                    
        """

        report_info = technique_reason_repo['clean']['numeric']['constant']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingConstant(constant, col_to_constant, data=self.data)

            if self.report is not None:
                if col_to_constant is None:
                    self.report.ReportTechnique(report_info, self.data.columns)
                else:
                    self.report.ReportTechnique(report_info, list(col_to_constant))

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = ReplaceMissingConstant(constant,
                                                                                                    col_to_constant,
                                                                                                    train_data=self.data_properties.train_data,
                                                                                                    test_data=self.data_properties.test_data)

            if self.report is not None:
                if col_to_constant is None:
                    self.report.ReportTechnique(report_info, self.data_properties.train_data.columns)
                else:
                    self.report.ReportTechnique(report_info, list(col_to_constant))

            return self.data_properties.train_data, self.data_properties.test_data

    def replace_missing_new_category(self, new_category=None, col_to_category=None):
        """
        Replaces missing values in categorical column with its own category. The categories can be autochosen
        from the defaults set.

        For numeric categorical columns default values are: -1, -999, -9999
        For string categorical columns default values are: "Other", "Unknown", "MissingDataCategory"

        This function exists in `clean/categorical.py` as `ReplaceMissingNewCategory`.
        
        Keyword Arguments:
            new_category {str} or  {int} or {float} -- Category to replace missing values with (default: {None})
            col_to_category {list} or {dict} -- A list of specific columns to apply this technique to or a dictionary
                                                mapping {Column Name: `constant`}. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                    Example: if custom_cols is provided and override is true, the technique will only be applied
                                    to the the columns in custom_cols (default: {False})

        Examples:

        >>>> ReplaceMissingCategory({'a': "Green", 'b': "Canada", 'c': "December"})
        >>>> ReplaceMissingCategory("Blue", ['a', 'b', 'c'])
        """
        
        report_info = technique_reason_repo['clean']['categorical']['new_category']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingNewCategory(constant=new_category, col_to_category=col_to_category, data=self.data)

            if self.report is not None:
                if col_to_category is None:
                    self.report.ReportTechnique(report_info, self.data.columns)
                else:
                    self.report.ReportTechnique(report_info, list(col_to_category))

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = ReplaceMissingNewCategory(constant=new_category,
                                                                                                        col_to_category=col_to_category,                                                                                                    
                                                                                                        train_data=self.data_properties.train_data,
                                                                                                        test_data=self.data_properties.test_data)

            if self.report is not None:
                if col_to_category is None:
                    self.report.ReportTechnique(report_info, self.data_properties.train_data.columns)
                else:
                    self.report.ReportTechnique(report_info, list(col_to_category))                                                                                                   

            return self.data_properties.train_data, self.data_properties.test_data


    def replace_missing_remove_row(self, cols_to_remove):
        """Remove rows where the value of a column for those rows is missing.

        This function exists in `clean/categorical.py` as `ReplaceMissingRemoveRow`.
        
        Keyword Arguments:
            cols_to_remove {list} -- A list of specific columns to remove. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                    Example: if custom_cols is provided and override is true, the technique will only be applied
                                    to the the columns in custom_cols (default: {False})
        """

        report_info = technique_reason_repo['clean']['categorical']['remove_rows']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingRemoveRow(cols_to_remove, data=self.data)

            if self.report is not None:
                self.report.ReportTechnique(report_info, cols_to_remove)

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = ReplaceMissingRemoveRow(cols_to_remove,                                                                                                    
                                                                                                    train_data=self.data_properties.train_data,
                                                                                                    test_data=self.data_properties.test_data)                                                                                        

            if self.report is not None:
                self.report.ReportTechnique(report_info, list(cols_to_remove))

            return self.data_properties.train_data, self.data_properties.test_data

    def remove_duplicate_rows(self, list_of_cols=[]):
        """
        Remove rows from the data that are exact duplicates of each other and leave only 1.
        This can be used to reduce processing time or performance for algorithms where
        duplicates have no effect on the outcome (i.e DBSCAN)
        
        This function exists in `clean/util.py` as `RemoveDuplicateRows`.
        
        Args:
            list_of_cols (list, optional): A list of specific columns to apply this technique to. Defaults to [].
        
        Returns:
            Dataframe, *Dataframe: Transformed dataframe with rows with a missing values in a specific column are missing
    
            * Returns 2 Dataframes if Train and Test data is provided.
        """
    
        report_info = technique_reason_repo['clean']['general']['remove_duplicate_rows']
    
        if self.data_properties.use_full_data:
            self.data = RemoveDuplicateRows(list_of_cols=list_of_cols, data=self.data)
    
            return self.data
    
        else:
            self.data_properties.train_data, self.data_properties.test_data = RemoveDuplicateRows(list_of_cols=[],
                                                                                                train_data=self.data_properties.train_data,
                                                                                                test_data=self.data_properties.test_data)

            return self.data_properties.train_data, self.data_properties.test_data

    def remove_duplicate_columns(self):
        """
        Remove columns from the data that are exact duplicates of each other and leave only 1.
        
        This function exists in `clean/util.py` as `RemoveDuplicateColumns`.
                
        Returns:
            Dataframe, *Dataframe: Transformed dataframe with rows with a missing values in a specific column are missing
    
            * Returns 2 Dataframes if Train and Test data is provided.
        """
    
        report_info = technique_reason_repo['clean']['general']['remove_duplicate_columns']
    
        if self.data_properties.use_full_data:
            self.data = RemoveDuplicateColumns(data=self.data)
    
            return self.data
    
        else:
            self.data_properties.train_data, self.data_properties.test_data = RemoveDuplicateColumns(train_data=self.data_properties.train_data,
                                                                                                test_data=self.data_properties.test_data)

            return self.data_properties.train_data, self.data_properties.test_data

    def GenerateCode(self):
        print("Not developed yet.")
        return
