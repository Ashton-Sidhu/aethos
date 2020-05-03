def replace_missing_fill(
    x_train, x_test=None, list_of_cols=[], method="", **extra_kwargs
):
    """
    Replaces missing values with the known values ahead of it and behind it.
    
    Parameters
    ----------
    x_train: Dataframe or array like - 2d
        Dataset
        
    x_test: Dataframe or array like - 2d
        Testing dataset, by default None.
        
    list_of_cols : list
        A list of specific columns to apply this technique to, by default []

    method : str
        Type of fill, by default ''
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific row are missing

    Returns 2 Dataframes if x_test is provided.
    """

    # Handle erroneous input
    extra_kwargs.pop("method", method)

    for col in list_of_cols:
        x_train[col] = x_train[col].fillna(method=method, **extra_kwargs)

        if x_test is not None:
            x_test[col] = x_test[col].fillna(method=method, **extra_kwargs)

    return x_train, x_test
