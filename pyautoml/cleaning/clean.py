import pandas as pd
import yaml

from pyautoml.cleaning.categorical import *
from pyautoml.cleaning.numeric import *
from pyautoml.cleaning.util import *
from pyautoml.data.data import Data
from pyautoml.util import GetListOfCols, SplitData, _FunctionInputValidation

with open("pyautoml/technique_reasons.yml", 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")

class Clean():

    
    def __init__(self, data=None, train_data=None, test_data=None, test_split_percentage=0.2, use_full_data=False, target_field="", report_name=None):        

        if not _FunctionInputValidation(data, train_data, test_data):
            print("Error initialzing constructor, please provide one of either data or train_data and test_data, not both.")

        self.data = data
        self.data_properties = Data(self.data, use_full_data=use_full_data, target_field="", report_name=report_name)

        if self.data is not None:
            self.train_data, self.test_data = SplitData(self.data, test_split_percentage)        
        else:
            self.train_data = self.data_properties.train_data
            self.test_data = self.data_properties.test_data

        if self.data_properties.report is None:
            self.reporting = False
        else:
            self.reporting = True
            self.report = self.data_properties.report
            self.report.WriteHeader("Cleaning")


    def RemoveColumns(self, threshold):
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
            new_columns = original_columns.difference(self.data.columns)
            self.report.ReportTechnique(report_info, new_columns)

            return self.data

        else:
            #Gather original data information
            original_columns = set(list(self.train_data.columns))

            self.data_properties.train_data, self.data_properties.test_data = RemoveColumns(threshold,
                                                                                            train_data=self.data_properties.train_data,
                                                                                            test_data=self.data_properties.test_data)

            new_columns = original_columns.difference(self.train_data.columns)
            self.report.ReportTechnique(report_info, new_columns)

            return self.data_properties.train_data, self.data_properties.test_data


    def RemoveRows(self, threshold):
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
            self.report.ReportTechnique(report_info, [])

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = RemoveRows(threshold,
                                                                                        train_data=self.data_properties.train_data,
                                                                                        test_data=self.data_properties.test_data)

            #Write to report            
            self.report.ReportTechnique(report_info, [])                                                                                    

            return self.data_properties.train_data, self.data_properties.test_data
    
    def ReplaceMissingMean(self, list_of_cols=[]):
        """Replaces missing values in every numeric column with the mean of that column.

        Mean: Average value of the column. Effected by outliers.

        This function exists in `clean/numeric.py` as `ReplaceMissingMeanMedianMode`.
       
        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the list_of_cols overrides the columns in field_types
                                    Example: if list_of_cols is provided and override is true, the technique will only be applied
                                    to the the columns in list_of_cols (default: {False})
        """
        #list_of_cols = GetListOfCols("numeric", list_of_cols, self.data_properties.field_types, override)

        report_info = technique_reason_repo['clean']['numeric']['mean']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingMeanMedianMode("mean", list_of_cols, data=self.data)

            #Write to report            
            self.report.ReportTechnique(report_info, list_of_cols)

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = ReplaceMissingMeanMedianMode("mean",
                                                                                                            list_of_cols=list_of_cols,
                                                                                                            train_data=self.data_properties.train_data,
                                                                                                            test_data=self.data_properties.test_data)

            self.report.ReportTechnique(report_info, list_of_cols)

            return self.data_properties.train_data, self.data_properties.test_data

    def ReplaceMissingMedian(self, list_of_cols=[]):
        """Replaces missing values in every numeric column with the median of that column.

        Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.

        This function exists in `clean/numeric.py` as `ReplaceMissingMeanMedianMode`.
        
        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the list_of_cols overrides the columns in field_types
                                    Example: if list_of_cols is provided and override is true, the technique will only be applied
                                    to the the columns in list_of_cols (default: {False})                       
        """

        #list_of_cols = GetListOfCols("numeric", list_of_cols, self.data_properties.field_types, override)
        report_info = technique_reason_repo['clean']['numeric']['median']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingMeanMedianMode("median", list_of_cols, data=self.data)

            self.report.ReportTechnique(report_info, list_of_cols)

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = ReplaceMissingMeanMedianMode("median",
                                                                                                            list_of_cols=list_of_cols,
                                                                                                            train_data=self.data_properties.train_data,
                                                                                                            test_data=self.data_properties.test_data)

            self.report.ReportTechnique(report_info, list_of_cols)

            return self.data_properties.train_data, self.data_properties.test_data

    def ReplaceMissingMode(self, list_of_cols=[]):
        """Replaces missing values in every numeric column with the mode of that column

        Mode: Most common number in a list of numbers.

        This function exists in `clean/numeric.py` as `ReplaceMissingMeanMedianMode`.
        
        Keyword Arguments:
            list_of_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the list_of_cols overrides the columns in field_types
                                    Example: if list_of_cols is provided and override is true, the technique will only be applied
                                    to the the columns in list_of_cols (default: {False})      
        """

        #list_of_cols = GetListOfCols("numeric", self.data_properties.field_types, override, list_of_cols)
        
        report_info = technique_reason_repo['clean']['numeric']['mode']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingMeanMedianMode("most_frequent", list_of_cols, data=self.data)

            self.report.ReportTechnique(report_info, list_of_cols)

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = ReplaceMissingMeanMedianMode("most_frequent",
                                                                                                            list_of_cols=list_of_cols,
                                                                                                            train_data=self.data_properties.train_data,
                                                                                                            test_data=self.data_properties.test_data)
            self.report.ReportTechnique(report_info, list_of_cols)

            return self.data_properties.train_data, self.data_properties.test_data

    def ReplaceMissingConstant(self, constant=0, col_to_constant=None):
        """Replaces missing values in every numeric column with a constant.

        This function exists in `clean/numeric.py` as `ReplaceMissingConstant`.
        
        Keyword Arguments:
            constant {int or float} -- Numeric value to replace all missing values with (default: {0})
            col_to_constant {list} or {dict} -- A list of specific columns to apply this technique to or a dictionary
            mapping {Column Name: `constant`}. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                    Example: if custom_cols is provided and override is true, the technique will only be applied
                                    to the the columns in custom_cols (default: {False})
        """

        #list_of_cols = GetListOfCols("numeric", self.data_properties.field_types, custom_cols, override)
        report_info = technique_reason_repo['clean']['numeric']['constant']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingConstant(constant, col_to_constant, data=self.data)

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

            if col_to_constant is None:
                self.report.ReportTechnique(report_info, self.data_properties.train_data.columns)
            else:
                self.report.ReportTechnique(report_info, list(col_to_constant))

            return self.data_properties.train_data, self.data_properties.test_data

    def ReplaceMissingNewCategory(self, new_category=None, col_to_category=None):
        """Replaces missing values in categorical column with its own category. The categories can be autochosen
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
        """
        
        #list_of_cols = GetListOfCols("categorical", self.data_properties.field_types, custom_cols, override)
        report_info = technique_reason_repo['clean']['categorical']['new_category']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingNewCategory(col_to_category=col_to_category, constant=new_category, data=self.data)

            if col_to_category is None:
                self.report.ReportTechnique(report_info, self.data.columns)
            else:
                self.report.ReportTechnique(report_info, list(col_to_category))

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = ReplaceMissingNewCategory(col_to_category=col_to_category,
                                                                                                    constant=new_category,
                                                                                                    train_data=self.data_properties.train_data,
                                                                                                    test_data=self.data_properties.test_data)

            if col_to_category is None:
                self.report.ReportTechnique(report_info, self.data_properties.train_data.columns)
            else:
                self.report.ReportTechnique(report_info, list(col_to_category))                                                                                                   

            return self.data_properties.train_data, self.data_properties.test_data


    def ReplaceMissingRemoveRow(self, cols_to_remove):
        """Remove rows where the value of a column for those rows is missing.

        This function exists in `clean/categorical.py` as `ReplaceMissingRemoveRow`.
        
        Keyword Arguments:
            cols_to_remove {list} -- A list of specific columns to remove. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                    Example: if custom_cols is provided and override is true, the technique will only be applied
                                    to the the columns in custom_cols (default: {False})
        """

        #list_of_cols = GetListOfCols("categorical", self.data_properties.field_types, cols_to_remove, override)

        report_info = technique_reason_repo['clean']['categorical']['remove_rows']

        if self.data_properties.use_full_data:
            self.data = ReplaceMissingRemoveRow(cols_to_remove, data=self.data)

            self.report.ReportTechnique(report_info, cols_to_remove)

            return self.data

        else:
            self.data_properties.train_data, self.data_properties.test_data = ReplaceMissingRemoveRow(cols_to_remove,                                                                                                    
                                                                                                    train_data=self.data_properties.train_data,
                                                                                                    test_data=self.data_properties.test_data)                                                                                        
        
            self.report.ReportTechnique(report_info, list(cols_to_remove))

            return self.data_properties.train_data, self.data_properties.test_data

    def GenerateCode(self):
        return
