import pandas as pd


def CheckMissingData(df):
        """
        Utility function that checks if the data has any missing values.

        Arguemnts:
            df {Dataframe} -- Dataframe of the data
                
        Returns:
            [Boolean] -- True if the data is missing values, False o/w.
        """
        return df.isnull().values.any()

def GetKeysByValues(dict_of_elements, value):
    """Utility function that returns the list of keys whos value matches a criteria defined
    by the param `value`.
    
    Arguments:
        dict {Dictionary} -- Dictionary of key value mapping
        value {any} -- Value you want to return keys of

    Returns:
        [list] -- List of keys whos value matches the criteria defined by the param `value`
    """
    return [key for (key, value) in dict_of_elements.items() if value == value]

def GetListOfCols(column_type, dict_of_values, override, custom_cols):
    """Utility function to get the list of columns based off their column type (numeric, str_categorical, num_categorical, text, etc.).
    If `custom_cols` is provided and override is True, then `custom_cols` will only be returned. If override is False then the filtered columns
    and the custom columns provided will be returned.
    
    Arguments:
        column_type {string} -- Type of the column - can be categorical, numeric, text or datetime
        dict_of_values {Dictionary} -- Dictionary of key-value pairs        
        custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
        override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                              Example: if custom_cols is provided and override is true, the technique will only be applied
                              to the the columns in custom_cols (default: {False})
    
    Returns:
        [list] -- list of columns matching the column_type criteria plus any custom columns specified or
                    just the columns specified in custom_cols if override is True
    """
    
    if override:
        list_of_cols = custom_cols
    else:
        list_of_cols = set(custom_cols + GetKeysByValues(dict_of_values, column_type))

    return list_of_cols

def DropAndReplaceColumns(df, drop_cols, new_data):
    """Utility function that drops a column that has been processed and replaces it with the new columns that have been derived from it.
    
    Arguments:
        df {Dataframe} -- Dataframe of the data
        drop_cols {str or [str]} -- column or columns to be dropped
        new_data {Dataframe} -- new data columns to be added to the dataframe
    
    Returns:
        [Dataframe] -- Dataframe with the dropped column and the new data added
    """

    df.drop(df[list_of_cols], inplace=True)
    df.concat(new_data, axis=1, inplace=True)
    return df
