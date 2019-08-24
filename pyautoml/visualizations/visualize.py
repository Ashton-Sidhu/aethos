import ptitprince as pt


def raincloud(col: str, target_col: str, data):


    params = {}


    ax = pt.RainCloud(x=target_col,
                    y=col,
                    data=data.infer_objects(),
                    pointplot=True,
                    width_viol=0.8,
                    width_box=.4,
                    figsize=(12,8),
                    orient='h',
                    move=0.
                    )
