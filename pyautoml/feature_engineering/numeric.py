import pandas as pd
from pyautoml.util import _numeric_input_conditions, drop_replace_columns
from sklearn.preprocessing import PolynomialFeatures


def polynomial_features(x_train, x_test=None, list_of_cols=[], **poly_kwargs):
    """
    Computes polynomial features from your existing features.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset
        
    x_test : DataFrame
        Testing dataset, by default None

    list_of_cols : list, optional
        A list of specific columns to apply this technique to
        If `list_of_cols` is not provided, the strategy will be
        applied to all numeric columns, by default []

    keep_col : bool, optional
        True to not remove the columns, by default False

    poly_kwargs : dict or kwargs
        Polynomial Features constructor key word arguments
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows normalized.

    Returns 2 Dataframes if x_test is provided.
    """

    poly = PolynomialFeatures(**poly_kwargs)
    list_of_cols = _numeric_input_conditions(list_of_cols, x_train)

    scaled_data = poly.fit_transform(x_train[list_of_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=poly.get_feature_names())
    x_train = drop_replace_columns(x_train, list_of_cols, scaled_df)
    
    if x_test is not None:
        scaled_x_test = poly.transform(x_test)
        scaled_test_df = pd.DataFrame(scaled_x_test, columns=poly.get_feature_names())
        x_test = drop_replace_columns(x_test, list_of_cols, scaled_test_df)

    return x_train, x_test
