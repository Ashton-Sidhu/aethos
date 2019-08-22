"""
This file contains the following functions:

apply
"""

from pyautoml.util import _function_input_validation


def apply(func, output_col: str, **datasets):
    """
    Wrapper for pandas apply function to be used in this library. Applies `func` to the entire data
    or just the trianing and testing data

    Parameters
    ----------
    func : Function pointer
            Function describing the transformation for the new column
    output_col : str
        New column name
        
    Either the full data or training data plus testing data MUST be provided, not both.

    data : DataFrame
        Full dataset, by default None
    train_data : DataFrame
        Training dataset, by default None
    test_data : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    * Returns 2 Dataframes if Train and Test data is provided. 
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError(f'Invalid parameters passed: {str(datasets)}')

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        data.loc[:, output_col] = data.apply(func, axis=1)

        return data
    else:
        train_data.loc[:, output_col] = train_data.apply(func, axis=1)
        test_data.loc[:, output_col] = test_data.apply(func, axis=1)

        return train_data, test_data
