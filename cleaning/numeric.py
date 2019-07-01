import pandas as pd
from sklearn.impute import SimpleImputer

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

        
        imp_mean = SimpleImputer(strategy='mean')
        list_of_cols = GetListOfCols(custom_cols, self.data_properties.field_types, override, "numeric")
        train_data = self.data_properties.train_data
        test_data = self.data_properties.test_data

        for col in cols:
            if self.data_properties.use_full_dataset:                
                fit_data = imp_mean.fit_transform(self.df[list_of_cols])
                self.df = DropAndReplaceColumns(self.df, list_of_cols, fit_data)
            else:
                fit_train_data = imp_mean.fit_transform(train_data)
                self.data_properties.train_data = DropAndReplaceColumns(train_data, list_of_cols, fit_train_data)
                
                fit_test_data = imp_mean.transform(test_data)
                self.data_properties.test_data = DropAndReplaceColumns(test_data, list_of_cols, fit_test_data)

    def ReplaceMissingMedian(self, custom_cols=[], override=False):
        """Replaces missing values in every numeric column with the median of that column.

        Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.
        
        Keyword Arguments:
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                  Example: if custom_cols is provided and override is true, the technique will only be applied
                                  to the the columns in custom_cols (default: {False})                       
        """

        imp_median = SimpleImputer(strategy='median')
        list_of_cols = GetListOfCols(custom_cols, self.data_properties.field_types, override, "numeric")
        train_data = self.data_properties.train_data
        test_data = self.data_properties.test_data

        for col in cols:
            if self.data_properties.use_full_dataset:                
                fit_data = imp_median.fit_transform(self.df[list_of_cols])
                self.df = DropAndReplaceColumns(self.df, list_of_cols, fit_data)
            else:
                fit_train_data = imp_median.fit_transform(train_data)
                self.data_properties.train_data = DropAndReplaceColumns(train_data, list_of_cols, fit_train_data)
                
                fit_test_data = imp_median.transform(test_data)
                self.data_properties.test_data = DropAndReplaceColumns(test_data, list_of_cols, fit_test_data)

    def ReplaceMissingMode(self, custom_cols=[], override=False):
        """Replaces missing values in every numeric column with the mode of that column

        Mode: Most common number in a list of numbers.
        
        Keyword Arguments:
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                  Example: if custom_cols is provided and override is true, the technique will only be applied
                                  to the the columns in custom_cols (default: {False})      
        """

        imp_mode = SimpleImputer(strategy='most_frequent')
        list_of_cols = GetListOfCols(custom_cols, self.data_properties.field_types, override, "numeric")
        train_data = self.data_properties.train_data
        test_data = self.data_properties.test_data

        for col in cols:
            if self.data_properties.use_full_dataset:                
                fit_data = imp_mode.fit_transform(self.df[list_of_cols])
                self.df = DropAndReplaceColumns(self.df, list_of_cols, fit_data)
            else:
                fit_train_data = imp_mode.fit_transform(train_data)
                self.data_properties.train_data = DropAndReplaceColumns(train_data, list_of_cols, fit_train_data)
                
                fit_test_data = imp_mode.transform(test_data)
                self.data_properties.test_data = DropAndReplaceColumns(test_data, list_of_cols, fit_test_data)

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

        for col in list_of_cols:
            self.df[col].fillna(constant, inplace=True)
            self.data_properties.train_data[col].fillna(constant, inplace=True)
            self.data_properties.test_data[col].fillna(constant, inplace=True)
