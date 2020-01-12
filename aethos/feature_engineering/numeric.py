import pandas as pd
from aethos.util import _numeric_input_conditions, drop_replace_columns
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


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


def drop_correlated_features(x_train, x_test=None, threshold=0.95):
    """
    Drops highly correlated features.

    Derived from: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Training dataset, by default None.
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.

    threshold : float, optional
        Correlation coefficient value threshold to drop the feature

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    Returns 2 Dataframes if x_test is provided.  
    """

    corr = x_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > 0.95)]

    x_train.drop(drop_cols, axis=1, inplace=True)

    if x_test is not None:
        x_test.drop(drop_cols, axis=1, inplace=True)

    return x_train, x_test
