import pandas as pd

from categorical import *
from numeric import *
from util import *


class Clean():

    def __init__(self, data, target_field=""):        
       self.df = data

    def RemoveColumns(self, threshold):
        """Remove columns from the dataframe that have more than the threshold value of missing columns.
        Example: Remove columns where > 50% of the data is missing
        
        Arguments:
            threshold {[float]} -- Value between 0 and 1 that describes what percentage of a column can be missing values.
        """

        self.df = RemoveColumns(self.df, threshold)    

    def RemoveRows(self, threshold):
        """Remove rows from the dataframe that have more than the threshold value of missing rows.
        Example: Remove rows where > 50% of the data is missing
        
        Arguments:
            threshold {[float]} -- Value between 0 and 1 that describes what percentage of a row can be missing values.
        """

        self.df = RemoveRows(self.df, threshold)
    
    def ReplaceMissingMean(self, custom_cols=[], override=False):
        """Replaces missing values in every numeric column with the mean of that column.

        Mean: Average value of the column. Effected by outliers.
        
        Keyword Arguments:
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                    Example: if custom_cols is provided and override is true, the technique will only be applied
                                    to the the columns in custom_cols (default: {False})
        """
        
        
        list_of_cols = GetListOfCols(custom_cols, self.data_properties.field_types, override, "numeric")
        
        self.df = ReplaceMissingMMM(list_of_cols, "mean", self.df)

    def ReplaceMissingMedian(self, custom_cols=[], override=False):
        """Replaces missing values in every numeric column with the median of that column.

        Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.
        
        Keyword Arguments:
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                    Example: if custom_cols is provided and override is true, the technique will only be applied
                                    to the the columns in custom_cols (default: {False})                       
        """

        list_of_cols = GetListOfCols(custom_cols, self.data_properties.field_types, override, "numeric")
        
        self.df = ReplaceMissingMMM(list_of_cols, "median", self.df)

    def ReplaceMissingMode(self, custom_cols=[], override=False):
        """Replaces missing values in every numeric column with the mode of that column

        Mode: Most common number in a list of numbers.
        
        Keyword Arguments:
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                    Example: if custom_cols is provided and override is true, the technique will only be applied
                                    to the the columns in custom_cols (default: {False})      
        """

        list_of_cols = GetListOfCols(custom_cols, self.data_properties.field_types, override, "numeric")
        
        self.df = ReplaceMissingMMM(list_of_cols, "most_frequent", self.df)

    def ReplaceMissingConstant(self, constant=0, custom_cols=[], override=False):
        """Replaces missing values in every numeric column with a constant.
        
        Keyword Arguments:
            constant {int or float} -- Numeric value to replace all missing values with (default: {0})
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                    Example: if custom_cols is provided and override is true, the technique will only be applied
                                    to the the columns in custom_cols (default: {False})
        """

        list_of_cols = GetListOfCols("numeric", self.data_properties.field_types, custom_cols, override)

        self.df = ReplaceMissingConstant(constant, list_of_cols, self.df)

    def ReplaceMissingNewCategory(self, new_category_name=None, custom_cols=[], override=False):
        """Replaces missing values in categorical column with its own category. The category name can be provided
        through the `new_category_name` parameter or if a category name is not provided, this function will assign 
        the name based off default categories.

        For numeric categorical columns default values are: -1, -999, -9999
        For string categorical columns default values are: "Other", "MissingDataCategory"
        
        Keyword Arguments:
            new_category_name {None, str, int, float} -- Category to replace missing values with (default: {None})
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                    Example: if custom_cols is provided and override is true, the technique will only be applied
                                    to the the columns in custom_cols (default: {False})
        """
        
        list_of_cols = GetListOfCols("numeric", self.data_properties.field_types, custom_cols, override)

        self.df = ReplaceMissingNewCategory(constant, list_of_cols, self.df)


    def ReplaceMissingRemoveRow(cols_to_remove=[], override=False):
        """Remove rows where the value of a column for those rows is missing.
        
        Keyword Arguments:
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                    Example: if custom_cols is provided and override is true, the technique will only be applied
                                    to the the columns in custom_cols (default: {False})
        """

        list_of_cols = GetListOfCols("numeric", self.data_properties.field_types, custom_cols, override)

        self.df = ReplaceMissingNewCategory(constant, list_of_cols, self.df)

    def GenerateCode(self):
        return
