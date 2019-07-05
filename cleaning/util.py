import pandas as pd


def RemoveColumns(df, threshold):
    """Remove columns from the dataframe that have more than the threshold value of missing columns.
    Example: Remove columns where > 50% of the data is missing
        
    Arguments:
        df {[DataFrame]} -- DataFrame of data
        threshold {[float]} -- Value between 0 and 1 that describes what percentage of a column can be missing values.

    Returns:
        [DataFrame] -- Dataframe with columns removed
    """
    
    criteria_meeting_columns = df.columns[df.isnull().mean() < threshold]
    
    return df[criteria_meeting_columns]

def RemoveRows(df, threshold):
    """Remove rows from the dataframe that have more than the threshold value of missing rows.
    Example: Remove rows where > 50% of the data is missing
    
    Arguments:
        df {[DataFrame]} -- DataFrame of data
        threshold {[float]} -- Value between 0 and 1 that describes what percentage of a row can be missing values.
        
    Returns:
        [DataFrame] -- Dataframe with columns removed
    """

    return df.dropna(thresh=round(df.shape[1] * threshold), axis=0)

def _FunctionInputValidation(data, train_data, test_data):
    """
    Helper function to help determine if input is valid.
    """

    if data is None and (train_data is None or test_data is None):
        return False

    if data is not None and (train_data is not None or test_data is not None):
        return False

    if train_data is not None and test_data is None:
        return False

    if test_data is not None and train_data is None:
        return False

    return True
