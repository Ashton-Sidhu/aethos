def get_input_types(df, target_field=""):
    """
    Credit: https://github.com/minimaxir/automl-gs/

    Get the input types for each field in the DataFrame that corresponds
    to an input type to be fed into the model.
    Valid values are ['text', 'categorical', 'numeric', 'datetime', 'ignore']
    
    Inputs:
        df -- A pandas DataFrame.
        target_field -- string indicating the target field, default empty string to allow for unsupervised learning.
    Returns:
        [Dictionary] -- A dict of {field_name: type} mappings.
    """

    fields = df.columns
    nrows = df.shape[0]
    avg_spaces = -1

    field_types = OrderedDict()

    for field in fields:
       
        field_type = df[field].dtype
        num_unique_values = df[field].nunique()
        if field_type == 'object':
            avg_spaces = df[field].str.count(' ').mean()

        # Automatically ignore `id`-related fields
        if field.lower() in ['id', 'uuid', 'guid', 'pk', 'name']:
            field_types[field] = 'ignore'

        # Datetime is a straightforward data type.
        elif field_type == 'datetime64[ns]':
            field_types[field] = 'datetime'

        # Assume a float is always numeric.
        elif field_type == 'float64':
            field_types[field] = 'numeric'

        # If it's an object where the contents has
        # many spaces on average, it's text
        elif field_type == 'object' and avg_spaces >= 2.0:
            field_types[field] = 'text'

        # If the field has very few distinct values, it's categorical
        elif  field_types == 'object' and avg_spaces < 2.0:
            field_types[field] = 'categorical'

        # If the field has many distinct integers, assume numeric.
        elif field_type == 'int64':
            field_types[field] = 'numeric'

        # If the field has many distinct nonintegers, it's not helpful.
        elif num_unique_values > 0.9 * nrows:
            field_types[field] = 'ignore'

        # The rest (e.g. bool) is categorical
        else:
            field_types[field] = 'categorical'

    # Print to console for user-level debugging
    print("Modeling with field specifications:")
    print("\n".join(["{}: {}".format(k, v) for k, v in field_types.items() if k != target_field]))

    field_types = {k: v for k, v in field_types.items() if v != 'ignore'}

    return field_types


def normalize_col_names(input_types):
    """
    Credit: https://github.com/minimaxir/automl-gs/

    Fixes unusual column names (e.g. Caps, Spaces)
    to make them suitable printing into code templates.

    Inputs:
        input_types -- dict of col names: input types
    Returns:
        Dictionary -- A dict of col names: input types with normalized keys
    """

    pattern = re.compile('\W+')
    fields = [(re.sub(pattern, '_', field.lower()), field, field_type)
                   for field, field_type in input_types.items()]

    return fields
