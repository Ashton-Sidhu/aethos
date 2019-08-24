import ptitprince as pt


def raincloud(col: str, target_col: str, data, params={}):
    """
    Visualizes 2 columns using raincloud.
    
    Parameters
    ----------
    col : str
        Column name of general data
    target_col : str
        Column name of measurable data, numerical
    data : Dataframe
        Dataframe of the data
    params: dict
        Parameters for the RainCloud visualization
    """

    if not params:
        params = {'x': col,
                'y': target_col,
                'data': data.infer_objects(),
                'pointplot': True,
                'width_viol': 0.8,
                'width_box': .4,
                'figsize': (12,8),
                'orient': 'h',
                'move': 0.
                }
    else:
        params = params

    ax = pt.RainCloud(**params)
