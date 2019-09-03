import pandas_bokeh
import ptitprince as pt


def raincloud(col: str, target_col: str, data, **params):
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
                
    ax = pt.RainCloud(**params)

def barplot(x, y, data, groupby=None, method=None, orient='v', stacked=False, **kwargs):
    """
    Visualizes a bar plot.

    Kwargs are bokeh plot, vbar, and hbar bokeh plots.
    
    Parameters
    ----------
    x : str
        Column name for the x axis.
    y : list
        Columns for the y axis
    data : Dataframe
        Dataset
    groupby : str
        Data to groupby - xaxis, optional, by default None
    method : str
        Method to aggregate groupy data
        Examples: min, max, mean, etc., optional
        by default None
    orient : str, optional
        Orientation of graph, 'h' for horizontal
        'v' for vertical, by default 'v'
    stacked : bool
            Whether to stack the different columns resulting in a stacked bar chart,
            by default False
    """
    
    data_copy = data[[x] + y].copy()
    data_copy = data_copy.set_index(x)

    if groupby:
        data_copy = data_copy.groupby(groupby, as_index=False)        
        data_copy = getattr(data_copy, method)()

    if orient == 'v':

        p_bar = data_copy.plot_bokeh.bar(
                                        figsize=(1500,500), # For now until auto scale is implemented
                                        stacked=stacked,
                                        alpha=0.6,
                                        **kwargs
                                        )
                        
    else:        

        p_bar = data_copy.plot_bokeh.barh(
                                        figsize=(1500,500),
                                        stacked=stacked,
                                        alpha=0.6,
                                        **kwargs                                        
                                        )
