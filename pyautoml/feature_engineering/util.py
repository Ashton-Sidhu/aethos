"""
This file contains the following functions:

apply
"""
def apply(x_train, func, output_col: str, x_test=None):
    """
    Wrapper for pandas apply function to be used in this library. Applies `func` to the entire data
    or just the trianing and testing data

    Parameters
    ----------
    x_train : DataFrame
        Dataset

    func : Function pointer
        Function describing the transformation for the new column

    output_col : str
        New column name
        
    x_test : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column

    Returns 2 Dataframes if x_test is provided. 
    """

    x_train.loc[:, output_col] = x_train.apply(func, axis=1)

    if x_test is not None:
        x_test.loc[:, output_col] = x_test.apply(func, axis=1)

    return x_train, x_test
