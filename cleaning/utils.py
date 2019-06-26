import re

import pandas as pd


class CleanBase():
    
    def __init__(self, data):
        self.field_types = {}
        self.df = data 

class CleanUtil(CleanBase):

    self.colMapping = {}

    def GetInputTypes(self, custom_cols={}, target_field=""):
        """
        Credit: https://github.com/minimaxir/automl-gs/

        Get the input types for each field in the DataFrame that corresponds
        to an input type to be fed into the model.
        Valid values are ['text', 'categorical', 'numeric', 'datetime', 'ignore']
        
        Inputs:            
            custom_cols -- A dictionary defining the column name as the key and the data type of that column,
                        possible values are: 'datetime', 'numeric', 'text', 'categorical'
            target_field -- string indicating the target field, default empty string to allow for unsupervised learning
        """

        #TODO: Improve this detection function and to detect numeric categorical data

        fields = self.df.columns
        nrows = self.df.shape[0]
        avg_spaces = -1
        id_field_substrings = ['id', 'uuid', 'guid', 'pk', 'name']

        for field in fields:
            if field in custom_cols:
                self.field_types[field] = custom_cols[field]
                continue

            field_type = self.df[field].dtype
            num_unique_values = self.df[field].nunique()

            if field_type == 'object':
                avg_spaces = self.df[field].str.count(' ').mean()
            
            # CHECK DATA TYPES FOR CATEGORIZATION            
            # Automatically ignore `id`-related fields
            if any(word in field.lower() for word in id_field_substrings):
                self.field_types[field] = 'ignore'

            # Datetime is a straightforward data type.
            elif field_type == 'datetime64[ns]':
                self.field_types[field] = 'datetime'

            # Assume a float is always numeric.
            elif field_type == 'float64':
                self.field_types[field] = 'numeric'

            # TODO: Figure out a better way to identify text that is not categorical
            # If it's an object where the content has many spaces on average, it's text
            elif field_type == 'object' and avg_spaces >= 2.0:
                self.field_types[field] = 'text'

            # TODO: Figure out a better way to identify text that is categorical
            # If the field has very few distinct values, it's categorical
            elif  self.field_types == 'object' and avg_spaces < 2.0:
                self.field_types[field] = 'categorical'

            # If the field has many distinct integers, assume numeric.
            elif field_type == 'int64':
                self.field_types[field] = 'numeric'
            
            else:
                self.field_types[field] = 'categorical'
            
            # CHECK DATA CHARACTERISTIC FOR CATEGORIZATION
            # If the field has many distinct nonintegers, it's not helpful.
            if num_unique_values > 0.9 * nrows and field_type == 'object':
                self.field_types[field] = 'ignore'

            # If the field has only 1 unique value, it does not tell us anything
            if num_unique_values == 1:
                self.field_types[field] = 'ignore'          
        
        # TODO: Log the input type classification as a report.
        # Print to console for user-level debugging
        print("Modeling with field specifications:")
        print("\n".join(["{}: {}".format(k, v) for k, v in self.field_types.items() if k != target_field]))

        self.field_types = {k: v for k, v in self.field_types.items() if v != 'ignore'}


    def NormalizeColNames(self):
        """
        Utility function that fixes unusual column names (e.g. Caps, Spaces)
        to make them suitable printing into code templates.        
        """
        new_column_names = {}
        pattern = re.compile('\W+')

        for name in self.df.columns:
            new_column_names[name] = re.sub(pattern, '_', name.lower())

                  
        self.colMapping = new_column_names
        self.df.rename(index=str, columns=new_column_names, inplace=True)

    def ReduceData(self):
        """
        Utility function that selects a subset of the data that has been categorized as a column worth feature engineering on.
        """
        self.df = self.df[list(self.field_types.keys())]

    def CheckMissingData(self):
        """
        Utility function that checks if the data has any missing values.
                
        Returns:
            [Boolean] -- True if the data is missing values, False o/w.
        """
        return self.df.isnull().values.any()

    def GetKeysByValues(self, dict_of_elements, value):
        """Utility function that returns the list of keys whos value matches a criteria
        
        Arguments:
            dict {Dictionary} -- Dictionary of key value mapping
            value {any} -- Value you want to return keys of
        """
        return [key for (key, value) in dict_of_elements.items() if value == value]
