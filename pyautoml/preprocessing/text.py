from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import PorterStemmer

from pyautoml.util import _function_input_validation

NLTK_STEMMERS : {
        'wordnet': WordNetLemmatizer(),
        'porter': PorterStemmer()
        }

def split_sentence(col_name: str, **datasets):
    """
    Splits text by its sentences and then saves that list in a new column.
    
    Parameters
    ----------
    col_name : str
        Column name of text data that you want to separate into sentences
    
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
        data.loc[:, col_name + '_sentences'] = data[col_name].apply(lambda x: sent_tokenize(x))

        return data
    else:
        train_data.loc[:, col_name + '_sentences'] = train_data[col_name].apply(lambda x: sent_tokenize(x))
        test_data.loc[:, col_name + '_sentences'] = test_data[col_name].apply(lambda x: sent_tokenize(x))

        return train_data, test_data

        
def nltk_stem(list_of_cols=[], stemmer='wordnet' **datasets):
    """
    Stems all the text in a column.
    
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
