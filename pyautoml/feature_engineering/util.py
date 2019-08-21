"""
This file contains the following functions:

apply
"""


def apply(func, output_col: str, **datasets):
    """
    [summary]
    
    Parameters
    ----------
    func : [type]
        [description]
    output_col : str
        [description]
    
    Returns
    -------
    [type]
        [description]
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError(f'Invalid parameters passed: {str(datasets)}')

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        data[output_col] = data.apply(func, axis=1)

        return data
    else:
        train_data[col] = train_data.apply(func, axis=1)
        test_data[col] = test_data.apply(func, axis=1)

        return train_data, test_data
