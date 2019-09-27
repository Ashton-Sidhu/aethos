"""
This file contains the following functions:

apply
"""

from pyautoml.util import _function_input_validation


def apply(func, output_col: str, **datasets):
    """
    Wrapper for pandas apply function to be used in this library. Applies `func` to the entire data
    or just the trianing and testing data

    Either the full data or training data plus testing data MUST be provided, not both.

    Parameters
    ----------
    func : Function pointer
        Function describing the transformation for the new column

    output_col : str
        New column name

    data : DataFrame
        Full dataset, by default None

    x_train : DataFrame
        Training dataset, by default None
        
    x_test : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column

    Returns 2 Dataframes if Train and Test data is provided. 
    """

    data = datasets.pop('data', None)
    x_train = datasets.pop('x_train', None)
    x_test = datasets.pop('x_test', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))
    if not _function_input_validation(data, x_train, x_test):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        data.loc[:, output_col] = data.apply(func, axis=1)

        return data
    else:
        x_train.loc[:, output_col] = x_train.apply(func, axis=1)
        x_test.loc[:, output_col] = x_test.apply(func, axis=1)

        return x_train, x_test
