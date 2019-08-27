import pandas_bokeh
import ptitprince as pt
from IPython.display import display


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

def barplot(x, y, data, x_label='', y_label='', title='', groupby=None, method=None, orient='v'):
        
    data_copy = data[[x,y]].copy()
    data_copy = data_copy.set_index(x)

    if groupby is None:

        p_bar = data_copy.plot_bokeh.bar(
                                        xlabel=x_label,
                                        ylabel=y_label,
                                        title=title,
                                        figsize=(1500,500)
                                        )
                        
    else:

        data_copy = data_copy.groupby(groupby, as_index=False)
        
        agg_func = getattr(data_copy, method)()

        p_bar = agg_func.plot_bokeh.bar(
                                        xlabel=x_label,
                                        ylabel=y_label,
                                        title=title,
                                        figsize=(1500,500)                                        
                                        )
