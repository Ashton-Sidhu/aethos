import os

from aethos.config.config import _global_config
from aethos.modelling.util import track_artifacts
from .model_analysis import ModelAnalysisBase


class UnsupervisedModelAnalysis(ModelAnalysisBase):
    def __init__(self, model, data, model_name):
        """
        Class to analyze Unsupervised models through metrics and visualizations.

        Parameters
        ----------
        model : str or Model Object
            Sklearn Model object or .pkl file of the object.

        data : pd.DataFrame
            Training Data used for the model.

        model_name : str
            Name of the model for saving images and model tracking purposes
        """

        self.model = model
        self.x_train = data
        self.model_name = model_name
        self.cluster_col = "predicted"

        if hasattr(self.model, "predict"):
            self.y_pred = self.model.predict(self.x_train)
        else:
            self.y_pred = self.model.fit_predict(self.x_train)

        self.x_train[self.cluster_col] = self.y_pred

    def filter_cluster(self, cluster_no: int):
        """
        Filters data by a cluster number for analysis.
        
        Parameters
        ----------
        cluster_no : int
            Cluster number to filter by
        
        Returns
        -------
        Dataframe
            Filtered data or test dataframe

        Examples
        --------
        >>> m = model.KMeans()
        >>> m.filter_cluster(1)
        """

        return self.x_train[self.x_train[self.cluster_col] == cluster_no]

    def plot_clusters(self, dim=2, reduce="pca", output_file="", **kwargs):
        """
        Plots the clusters in either 2d or 3d space with each cluster point highlighted
        as a different colour.

        For 2d plotting options, see:
        
            https://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.scatter

            https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#userguide-styling-line-properties 

        For 3d plotting options, see:

            https://www.plotly.express/plotly_express/#plotly_express.scatter_3d
            
        Parameters
        ----------
        dim : 2 or 3, optional
            Dimension of the plot, either 2 for 2d, 3 for 3d, by default 2

        reduce : str {'pca', 'tvsd', 'lle', 'tsne'}, optional
            Dimension reduction strategy i.e. pca, by default "pca"

        output_file: str
            Output file name including extension (.png, .jpg, etc.) to save image as.

        Examples
        --------
        >>> m = model.KMeans()
        >>> m.plot_clusters()
        >>> m.plot_clusters(dim=3)
        """

        output_file_path = (
            os.path.join(self.model_name, output_file) if output_file else output_file
        )

        self.plot_dim_reduction(
            self.cluster_col,
            dim=dim,
            algo=reduce,
            output_file=output_file_path,
            **kwargs,
        )

        if _global_config["track_experiments"]:  # pragma: no cover
            track_artifacts(self.run_id, self.model_name)
