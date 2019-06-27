import pandas as pd

from base import CleanBase
from data.util import GetListOfCols


class CleanNumeric(CleanBase):

    #TODO: Implement KNN, Interpolation, Extrapolation, Hot-Deck imputation for replacing missing data
    #TODO: Add data validation on the custom_cols argument to make sure it is float or int

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

        for col in list_of_cols:
            self.df[col].fillna(self.df[col].mean()[0], inplace=True)

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

        for col in list_of_cols:
            self.df[col].fillna(self.df[col].median()[0], inplace=True)

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
       
        for col in list_of_cols:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)

    def ReplaceMissingConstant(self, constant=0, custom_cols=[] override=False):
        """Replaces missing values in every numeric column with a constant.
        
        Keyword Arguments:
            constant {int or float} -- Numeric value to replace all missing values with (default: {0})
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                  Example: if custom_cols is provided and override is true, the technique will only be applied
                                  to the the columns in custom_cols (default: {False})
        """

        list_of_cols = GetListOfCols(custom_cols, self.data_properties.field_types, override, "numeric")

        for col in list_of_cols:
            self.df[col].fillna(constant, inplace=True)
