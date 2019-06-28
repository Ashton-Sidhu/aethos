import re

import pandas as pd
from sklearn.model_selection import train_test_split


class Data():    

    def __init__(self, data, target_field):

        self.orig_data = data
        self.field_types = {}
        self.colMapping = {}
        self.target_field = target_field
        self.train_data = None
        self.test_data = None

    def GetInputTypes(self, df, custom_cols={}):
        """
        Credit: https://github.com/minimaxir/automl-gs/

        Get the input types for each field in the DataFrame that corresponds
        to an input type to be fed into the model.
        Valid values are ['text', 'categorical', 'numeric', 'datetime', 'ignore']
        
        Arguemnts:
            df {Dataframe} -- Dataframe of the data            
            custom_cols {list} -- A dictionary defining the column name as the key and the data type of that column,
                        possible values are: 'datetime', 'numeric', 'text', 'categorical'
        """

        #TODO: Improve this detection function and to detect numeric categorical data

        fields = df.columns
        nrows = df.shape[0]
        avg_spaces = -1
        id_field_substrings = ['id', 'uuid', 'guid', 'pk', 'name']

        for field in fields:
            if field in custom_cols:
                self.field_types[field] = custom_cols[field]
                continue

            field_type = df[field].dtype
            num_unique_values = df[field].nunique()

            if field_type == 'object':
                avg_spaces = df[field].str.count(' ').mean()
            
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
                self.field_types[field] = 'str_categorical'

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
        print("\n".join(["{}: {}".format(k, v) for k, v in self.field_types.items() if k != self.target_field]))

        self.field_types = {k: v for k, v in self.field_types.items() if v != 'ignore'}


    def NormalizeColNames(self, df):
        """
        Utility function that fixes unusual column names (e.g. Caps, Spaces)
        to make them suitable printing into code templates.

        Arguemnts:
            df {Dataframe} -- Dataframe of the data

        Returns:
            [Dataframe] -- Dataframe whos column names have been normalized.       
        """
        new_column_names = {}
        pattern = re.compile('\W+')

        for name in df.columns:
            new_column_names[name] = re.sub(pattern, '_', name.lower())
                  
        self.colMapping = new_column_names
        return df.rename(index=str, columns=new_column_names, inplace=True)

    def ReduceData(self, df):
        """
        Utility function that selects a subset of the data that has been categorized as a column worth feature engineering on.

        Arguments:
            df {Dataframe} -- Dataframe of the data
        
        Returns:
            [Dataframe] -- Subset of the dataframe
        """
        return df[list(self.field_types.keys())]

    
    def StandardizeData(self, df, custom_cols={}):
        """Standarizes the properties of the dataset: column names and removes unimportant columns.
        Initializes the types of each column (categorical, numeric, etc.)
        
        Arguments:
            df {Dataframe} -- Dataframe of the data
        
        Keyword Arguments:
            custom_cols {dict} -- Mapping of column name to a column type (numeric, str_categorical, num_categorical, text, etc.)  (default: {{}})
        
        Returns:
            [Dataframe] -- Standardized version of the dataframe
        """

        df = self.NormalizeColNames(df)
        self.GetInputTypes(df, custom_cols)
        df = self.ReduceData(df)
        self.standardized = True

        return df

    def SplitData(self, split_percentage):

        self.train, self.test = train_test_split(df, test_size=split_percentage)
