import ptitprince as pt

import pandas_bokeh


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

def barplot(x, y, data, groupby=None, method=None, orient='v', **kwargs):
    """
    Visualizes a bar plot.
    
    Parameters
    ----------
    x : str
        Column name for the x axis.
    y : str
        Column for the y axis
    data : Dataframe
        [description]
    groupby : str
        Data to groupby - xaxis, optional, by default None
    method : str
        Method to aggregate groupy data
        Examples: min, max, mean, etc., optional
        by default None
    orient : str, optional
        Orientation of graph, 'h' for horizontal
        'v' for vertical, by default 'v'
    """
        
    data_copy = data[[x,y]].copy()
    data_copy = data_copy.set_index(x)

    if groupby:
        data_copy = data_copy.groupby(groupby, as_index=False)        
        data_copy = getattr(data_copy, method)()

    if orient == 'v':

        p_bar = data_copy.plot_bokeh.bar(
                                        figsize=(1500,500), # For now until auto scale is implemented
                                        **kwargs
                                        )
                        
    else:        

        p_bar = data_copy.plot_bokeh.barh(
                                        figsize=(1500,500),
                                        **kwargs                                        
                                        )
