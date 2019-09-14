from sklearn.preprocessing import LabelEncoder


def label_encoder(list_of_cols=[], **datasets):
    """
    Label encodes the columns provided.
    
    Either the full data or training data plus testing data MUST be provided, not both.
    
    Parameters
    ----------
    list_of_cols : list, optional
        A list of specific columns to apply this technique to
        If `list_of_cols` is not provided, the strategy will be
        applied to all numeric columns., by default []
    data: Dataframe or array like - 2d
        Full dataset, by default None.
    train_data: Dataframe or array like - 2d
        Training dataset, by default None.
    test_data: Dataframe or array like - 2d
        Testing dataset, by default None.

    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with rows with a missing values in a specific column are missing

    Returns 2 Dataframes if Train and Test data is provided.  
    """

    data = datasets.pop('data', None)
    train_data = datasets.pop('train_data', None)
    test_data = datasets.pop('test_data', None)

    if datasets:
        raise TypeError("Invalid parameters passed: {}".format(str(datasets)))

    if not _function_input_validation(data, train_data, test_data):
        raise ValueError("Function input is incorrectly provided.")
    
    label_encode = LabelEncoder()

    if data is not None:                
        fit_data = label_encode.fit_transform(data[list_of_cols])
        fit_df = pd.DataFrame(fit_data, columns=list_of_cols)
        data = drop_replace_columns(data, list_of_cols, fit_df)

        return data
    else:
        fit_train_data = label_encode.fit_transform(train_data[list_of_cols])
        fit_train_df = pd.DataFrame(fit_train_data, columns=list_of_cols)            
        train_data = drop_replace_columns(train_data, list_of_cols, fit_train_df)
        
        fit_test_data = label_encode.transform(test_data[list_of_cols])
        fit_test_df = pd.DataFrame(fit_test_data, columns=list_of_cols)      
        test_data = drop_replace_columns(test_data, list_of_cols, fit_test_df)

        return train_data, test_data
