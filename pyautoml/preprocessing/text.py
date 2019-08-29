from nltk import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from pyautoml.util import _function_input_validation


def split_sentence(list_of_cols=[], **datasets):
    """
    Splits text by its sentences and then saves that list in a new column.
    
    Parameters
    ----------
    list_of_cols : list, optional
        Column name(s) of text data that you want to separate into sentences
    
    Either the full data or training data plus testing data MUST be provided, not both.

    data : DataFrame
        Full dataset, by default None
    train_data : DataFrame
        Training dataset, by default None
    test_data : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe
        Transformed dataframe with the new column
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError(f'Invalid parameters passed: {str(datasets)}')

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    if data is not None:
        for col in list_of_cols:
            data.loc[:, col + '_sentences'] = data[col].apply(lambda x: sent_tokenize(x))

        return data
    else:
        for col in list_of_cols:
            train_data.loc[:, col + '_sentences'] = train_data[col].apply(lambda x: sent_tokenize(x))
            test_data.loc[:, col + '_sentences'] = test_data[col].apply(lambda x: sent_tokenize(x))

        return train_data, test_data

def lemmatize(list_of_cols=[], **datasets):
    """
    Lemmatizes all the text in a column.
    
    Parameters
    ----------
    list_of_cols : list, optional
        [description], by default []
    
    Parameters
    ----------
    list_of_cols : list, optional
        Column name(s) of text data that you want to separate into sentences
    
    Either the full data or training data plus testing data MUST be provided, not both.

    data : DataFrame
        Full dataset, by default None
    train_data : DataFrame
        Training dataset, by default None
    test_data : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe
        Transformed dataframe with the new column
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError(f'Invalid parameters passed: {str(datasets)}')

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError('Function input is incorrectly provided.')

    Initialize variables here

    if data is not None:
        for col in list_of_cols:
            data[col + '_clean'] = []

        return data
    else:
        for col in list_of_cols:

        

        return train_data, test_data
