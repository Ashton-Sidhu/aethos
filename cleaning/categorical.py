import pandas as pd

from utils import *


class CleanCategorical(CleanUtil):

    #TODO: Implement KNN, and replacing with most common category 

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
        
        str_missing_categories = ["Other", "MissingDataCategory"]
        num_missing_categories = [-1, -999, -9999]

        if isinstance(new_category_name, (None, str, int, float):
            print("Replace category is an invadlid type.")
            return

        if new_category_name is None:

            list_of_cols = self.GetListOfCols(custom_cols, override, "categorical")
            
            for col in list_of_cols:
                #If column is categorical string, replace NaNs with default string category from str_missing_categories
                if self.field_types[col] is "str_categorical":
                    #Determine default missing category for string
                    new_category_name = self._DetermineDefaultCategory(col, str_missing_categories)

                if self.field_types[col] is "num_categorical":
                    #Determine default missing category for numeric columns
                    new_category_name = self._DetermineDefaultCategory(col, num_missing_categories)

                self.df[col].fillna(new_category_name, inplace=True)

        else:

            if isinstance(new_category_name, (int, float)):
                list_of_cols = self.GetListOfCols(custom_cols, override, "num_categorical")
                
                for col in list_of_cols:
                    self.df[col].fillna(new_category_name, inplace=True)

            if isinstance(new_category_name, str):
                list_of_cols = self.GetListOfCols(custom_cols, override, "str_categorical")
                
                for col in list_of_cols:
                    self.df[col].fillna(new_category_name, inplace=True)
        

    def ReplaceMissingRemoveRow(self, custom_cols=[], override=False):
        return

    def _DetermineDefaultCategory(self, col, replacement_categories):
        """A utility function to help determine the default category name for a column that has missing
        categorical values. It takes in a list of possible values and if any the first value in the list
        that is not a value in the column is the category that will be used to replace missing values.
        
        Arguments:
            col {string} -- Column of the dataframe
            replacement_categories {list} -- List of potential category names to replace missing values with
        
        Returns:
            [type of contents of replacement_categories (str, int, etc.)] -- the default category for that column 
        """

        unique_vals_col = self.df[col].unique()
        for potential_category in replacement_categories:

            #If the potential category is not already a category, it becomes the default missing category 
            if potential_category not in unique_vals_col:
                new_category_name = potential_category
                break

        return new_category_name
