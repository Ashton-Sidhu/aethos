import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD


def sklearn_dim_reduction(
    x_train, x_test=None, algo=None, n_components=50, **dim_reduce_kwargs
):
    """
    Performs Principal Component Analysis on a dataset.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset

    n_components : Any
        Number of components to reduce to - algo dependent.

    x_test : DataFrame
        Testing dataset, by default None
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column

    Returns 2 Dataframes if x_test is provided. 
    """

    algorithms = {
        "pca": PCA(n_components=n_components, **dim_reduce_kwargs),
        "tsvd": TruncatedSVD(n_components=n_components, **dim_reduce_kwargs),
    }

    reducer = algorithms[algo]

    x_train = pd.DataFrame(reducer.fit_transform(x_train))
    x_train.columns = map(str, x_train.columns)

    if x_test is not None:
        x_test = pd.DataFrame(reducer.transform(x_test))
        x_test.columns = map(str, x_test.columns)

    return x_train, x_test
