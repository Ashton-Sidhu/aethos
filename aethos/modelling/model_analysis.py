import itertools
import math
import os
import warnings
from collections import OrderedDict
from itertools import compress

import catboost as cb
import interpret
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import sklearn
import xgboost as xgb
from aethos.config.config import _global_config
from aethos.feature_engineering.util import pca
from aethos.modelling.constants import (
    CLASS_METRICS_DESC,
    INTERPRET_EXPLAINERS,
    PROBLEM_TYPE,
    REG_METRICS_DESC,
    SHAP_LEARNERS,
)
from aethos.modelling.model_explanation import MSFTInterpret, Shap
from aethos.modelling.util import to_pickle
from aethos.templates.template_generator import TemplateGenerator as tg
from aethos.visualizations.util import _make_image_dir
from aethos.visualizations.visualize import *
from graphviz import Source
from IPython.display import HTML, SVG, display


class ModelBase(object):

    # TODO: Add more SHAP use cases

    def __init__(self, model_object, model, model_name, **kwargs):

        self.model = model
        self.model_name = model_name
        self.x_train = model_object.x_train
        self.x_test = model_object.x_test
        self.report = model_object.report

        shap_values = kwargs.pop("shap_values", None)

        if isinstance(self, ClassificationModel) or isinstance(self, RegressionModel):
            self.shap = Shap(
                self.model,
                self.x_train,
                self.x_test,
                self.y_test,
                SHAP_LEARNERS[type(self.model)],
                shap_values,
            )
            self.interpret = MSFTInterpret(
                self.model,
                self.x_train,
                self.x_test,
                self.y_test,
                PROBLEM_TYPE[type(self.model)],
            )
        else:
            self.shap = None
            self.interpret = None

        for method in dir(self.model):
            try:
                if not method.startswith("_") and not method.startswith("predict"):
                    self.__setattr__(method, getattr(self.model, method))

            except AttributeError as e:
                continue

    def model_weights(self):
        """
        Prints and logs all the features ranked by importance from most to least important.
        
        Returns
        -------
        dict
            Dictionary of features and their corresponding weights
        
        Raises
        ------
        AttributeError
            If model does not have coefficients to display

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.model_weights()
        """

        report_strings = []

        try:
            model_dict = dict(zip(self.features, self.model.coef_.flatten()))
        except Exception as e:
            raise AttributeError("Model does not have coefficients to view.")

        sorted_features = OrderedDict(
            sorted(model_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )

        for feature, weight in sorted_features.items():
            report_string = "\t{} : {:.2f}".format(feature, weight)
            report_strings.append(report_string)

            print(report_string.strip())

        if self.report:
            self.report.log("Features ranked from most to least important:\n")
            self.report.write_contents("\n".join(report_strings))

        return sorted_features

    def summary_plot(self, output_file="", **summaryplot_kwargs):
        """
        Create a SHAP summary plot, colored by feature values when they are provided.

        For a list of all kwargs please see the Shap documentation : https://shap.readthedocs.io/en/latest/#plots

        Parameters
        ----------
        output_file: str
            Output file name including extension (.png, .jpg, etc.) to save image as.

        max_display : int
            How many top features to include in the plot (default is 20, or 7 for interaction plots), by default None
            
        plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin", or "compact_dot"
            What type of summary plot to produce. Note that "compact_dot" is only used for SHAP interaction values.

        color : str or matplotlib.colors.ColorMap 
            Color spectrum used to draw the plot lines. If str, a registered matplotlib color name is assumed.

        axis_color : str or int 
            Color used to draw plot axes.

        title : str 
            Title of the plot.

        alpha : float 
            Alpha blending value in [0, 1] used to draw plot lines.

        show : bool 
            Whether to automatically display the plot.

        sort : bool
            Whether to sort features by importance, by default True

        color_bar : bool 
            Whether to draw the color bar.

        auto_size_plot : bool 
            Whether to automatically size the matplotlib plot to fit the number of features displayed. If False, specify the plot size using matplotlib before calling this function.

        layered_violin_max_num_bins : int
            Max number of bins, by default 20

        **summaryplot_kwargs
            For more info see https://shap.readthedocs.io/en/latest/#plots

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.summary_plot()
        """

        if self.shap is None:
            raise NotImplementedError(
                f"SHAP is not implemented yet for {str(type(self))}"
            )

        self.shap.summary_plot(output_file=output_file, **summaryplot_kwargs)

        if output_file and self.report:
            self.report.write_image(output_file)

    def decision_plot(
        self,
        num_samples=0.6,
        sample_no=None,
        highlight_misclassified=False,
        output_file="",
        **decisionplot_kwargs,
    ):
        """
        Visualize model decisions using cumulative SHAP values.
        
        Each colored line in the plot represents the model prediction for a single observation. 
        
        Note that plotting too many samples at once can make the plot unintelligible.

        When is a decision plot useful:        
            - Show a large number of feature effects clearly.
            
            - Visualize multioutput predictions.
            
            - Display the cumulative effect of interactions.
            
            - Explore feature effects for a range of feature values.
            
            - Identify outliers.
            
            - Identify typical prediction paths.
            
            - Compare and contrast predictions for several models.

        Explanation:
            - The plot is centered on the x-axis at the models expected value.

            - All SHAP values are relative to the model's expected value like a linear model's effects are relative to the intercept.
            
            - The y-axis lists the model's features. By default, the features are ordered by descending importance. 
            
            - The importance is calculated over the observations plotted. This is usually different than the importance ordering for the entire dataset. In addition to feature importance ordering, the decision plot also supports hierarchical cluster feature ordering and user-defined feature ordering.
            
            - Each observation's prediction is represented by a colored line. 
            
            - At the top of the plot, each line strikes the x-axis at its corresponding observation's predicted value. This value determines the color of the line on a spectrum.
            
            - Moving from the bottom of the plot to the top, SHAP values for each feature are added to the model's base value. This shows how each feature contributes to the overall prediction.
            
            - At the bottom of the plot, the observations converge at the models expected value.

        Parameters
        ----------
        output_file: str
            Output file name including extension (.png, .jpg, etc.) to save image as.

        num_samples : int, float, or 'all', optional
            Number of samples to display, if less than 1 it will treat it as a percentage, 'all' will include all samples
            , by default 0.6

        sample_no : int, optional
            Sample number to isolate and analyze, if provided it overrides num_samples, by default None

        highlight_misclassified : bool, optional
            True to highlight the misclassified results, by default False

        feature_order : str or None or list or numpy.ndarray
            Any of "importance" (the default), "hclust" (hierarchical clustering), "none", or a list/array of indices.
            hclust is useful for finding outliers.

        feature_display_range: slice or range
            The slice or range of features to plot after ordering features by feature_order. A step of 1 or None will display the features in ascending order. A step of -1 will display the features in descending order. If feature_display_range=None, slice(-1, -21, -1) is used (i.e. show the last 20 features in descending order). If shap_values contains interaction values, the number of features is automatically expanded to include all possible interactions: N(N + 1)/2 where N = shap_values.shape[1].

        highlight : Any 
            Specify which observations to draw in a different line style. All numpy indexing methods are supported. For example, list of integer indices, or a bool array.

        link : str 
            Use "identity" or "logit" to specify the transformation used for the x-axis. The "logit" link transforms log-odds into probabilities.

        plot_color : str or matplotlib.colors.ColorMap 
            Color spectrum used to draw the plot lines. If str, a registered matplotlib color name is assumed.

        axis_color : str or int 
            Color used to draw plot axes.

        y_demarc_color : str or int 
            Color used to draw feature demarcation lines on the y-axis.

        alpha : float 
            Alpha blending value in [0, 1] used to draw plot lines.

        color_bar : bool 
            Whether to draw the color bar.

        auto_size_plot : bool 
            Whether to automatically size the matplotlib plot to fit the number of features displayed. If False, specify the plot size using matplotlib before calling this function.

        title : str 
            Title of the plot.

        xlim: tuple[float, float] 
            The extents of the x-axis (e.g. (-1.0, 1.0)). If not specified, the limits are determined by the maximum/minimum predictions centered around base_value when link='identity'. When link='logit', the x-axis extents are (0, 1) centered at 0.5. x_lim values are not transformed by the link function. This argument is provided to simplify producing multiple plots on the same scale for comparison.

        show : bool 
            Whether to automatically display the plot.

        return_objects : bool 
            Whether to return a DecisionPlotResult object containing various plotting features. This can be used to generate multiple decision plots using the same feature ordering and scale, by default True.

        ignore_warnings : bool 
            Plotting many data points or too many features at a time may be slow, or may create very large plots. Set this argument to True to override hard-coded limits that prevent plotting large amounts of data.

        new_base_value : float 
            SHAP values are relative to a base value; by default, the expected value of the model's raw predictions. Use new_base_value to shift the base value to an arbitrary value (e.g. the cutoff point for a binary classification task).

        legend_labels : list of str 
            List of legend labels. If None, legend will not be shown.

        legend_location : str 
            Legend location. Any of "best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center".
    
        Returns
        -------
        DecisionPlotResult 
            If return_objects=True (the default). Returns None otherwise.

        Examples
        --------
        >>> # Plot two decision plots using the same feature order and x-axis.
        >>> m = model.LogisticRegression()
        >>> r = m.decision_plot()
        >>> m.decision_plot(no_sample=42, feature_order=r.feature_idx, xlim=r.xlim)
        """

        if self.shap is None:
            raise NotImplementedError(
                f"SHAP is not implemented yet for {str(type(self))}"
            )

        if highlight_misclassified:
            if not any(self.shap.misclassified_values):
                raise AttributeError("There are no misclassified values!")

            decisionplot_kwargs["highlight"] = self.shap.misclassified_values

        dp = self.shap.decision_plot(
            num_samples, sample_no, output_file=output_file, **decisionplot_kwargs
        )

        if output_file and self.report:
            self.report.write_image(output_file)

        return dp

    def force_plot(
        self, sample_no=None, misclassified=False, output_file="", **forceplot_kwargs
    ):
        """
        Visualize the given SHAP values with an additive force layout
        
        Parameters
        ----------
        sample_no : int, optional
            Sample number to isolate and analyze, by default None

        misclassified : bool, optional
            True to only show the misclassified results, by default False

        output_file: str
            Output file name including extension (.png, .jpg, etc.) to save image as.

        link : "identity" or "logit"
            The transformation used when drawing the tick mark labels. Using logit will change log-odds numbers
            into probabilities. 

        matplotlib : bool
            Whether to use the default Javascript output, or the (less developed) matplotlib output. Using matplotlib
            can be helpful in scenarios where rendering Javascript/HTML is inconvenient. 
        
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.force_plot() # The entire test dataset
        >>> m.forceplot(no_sample=1, misclassified=True) # Analyze the first misclassified result
        """

        if self.shap is None:
            raise NotImplementedError(
                f"SHAP is not implemented yet for {str(type(self))}"
            )

        if misclassified:
            if not any(self.shap.misclassified_values):
                raise AttributeError("There are no misclassified values!")

            forceplot_kwargs["shap_values"] = self.shap.shap_values[
                self.shap.misclassified_values
            ]

        fp = self.shap.force_plot(
            sample_no, output_file=output_file, **forceplot_kwargs
        )

        if output_file and self.report:
            self.report.write_image(output_file)

        return fp

    def dependence_plot(
        self, feature: str, interaction="auto", output_file="", **dependenceplot_kwargs
    ):
        """
        A dependence plot is a scatter plot that shows the effect a single feature has on the predictions made by the mode.

        Explanation:
            - Each dot is a single prediction (row) from the dataset.
        
            - The x-axis is the value of the feature (from the X matrix).

            - The y-axis is the SHAP value for that feature, which represents how much knowing that feature's value changes the output of the model for that sample's prediction.

            - The color corresponds to a second feature that may have an interaction effect with the feature we are plotting (by default this second feature is chosen automatically). 
            
            - If an interaction effect is present between this other feature and the feature we are plotting it will show up as a distinct vertical pattern of coloring. 
        
        Parameters
        ----------
        feature : str
            Feature who's impact on the model you want to analyze

        interaction : "auto", None, int, or string
            The index of the feature used to color the plot. The name of a feature can also be passed as a string. If "auto" then shap.common.approximate_interactions is used to pick what seems to be the strongest interaction (note that to find to true stongest interaction you need to compute the SHAP interaction values).
        
        output_file: str
            Output file name including extension (.png, .jpg, etc.) to save image as.

        x_jitter : float (0 - 1)
            Adds random jitter to feature values. May increase plot readability when feature is discrete.

        alpha : float
            The transparency of the data points (between 0 and 1). This can be useful to the show density of the data points when using a large dataset.

        xmin : float or string
            Represents the lower bound of the plot's x-axis. It can be a string of the format "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

        xmax : float or string
            Represents the upper bound of the plot's x-axis. It can be a string of the format "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

        ax : matplotlib Axes object
            Optionally specify an existing matplotlib Axes object, into which the plot will be placed. In this case we do not create a Figure, otherwise we do.

        cmap : str or matplotlib.colors.ColorMap 
            Color spectrum used to draw the plot lines. If str, a registered matplotlib color name is assumed.

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.dependence_plot()
        """

        if self.shap is None:
            raise NotImplementedError(
                f"SHAP is not implemented yet for {str(type(self))}"
            )

        dp = self.shap.dependence_plot(
            feature, interaction, output_file=output_file, **dependenceplot_kwargs
        )

        if output_file and self.report:
            self.report.write_image(output_file)

        return dp

    def shap_get_misclassified_index(self):
        """
        Prints the sample numbers of misclassified samples.

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.shap_get_misclassified_index()
        """

        sample_list = list(
            compress(
                range(len(self.shap.misclassified_values)),
                self.shap.misclassified_values,
            )
        )

        print(", ".join(str(np.array(sample_list) + 1)))

    def interpret_model(self, show=True):  # pragma: no cover
        """
        Displays a dashboard interpreting your model's performance, behaviour and individual predictions.

        If you have run any other `interpret` functions, they will be included in the dashboard, otherwise all the other intrepretable methods will be included in the dashboard.

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.interpret_model()
        """

        warnings.simplefilter("ignore")

        if isinstance(self.model, xgb.XGBModel):
            return "Using MSFT interpret is currently unsupported with XGBoost."

        if show:
            self.interpret.create_dashboard()

    def interpret_model_performance(
        self, method="all", predictions="default", show=True, **interpret_kwargs
    ):
        """
        Plots an interpretable display of your model based off a performance metric.

        Can either be 'ROC' or 'PR' for precision, recall for classification problems.

        Can be 'regperf' for regression problems.

        If 'all' a dashboard is displayed with the corresponding explainers for the problem type.

        ROC: Receiver Operator Characteristic
        PR: Precision Recall
        regperf: RegeressionPerf
        
        Parameters
        ----------
        method : str
            Performance metric, either 'all', 'roc' or 'PR', by default 'all'

        predictions : str, optional
            Prediction type, can either be 'default' (.predict) or 'probability' if the model can predict probabilities, by default 'default'

        show : bool, optional 
            False to not display the plot, by default True

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.interpret_model_performance()
        """

        warnings.simplefilter("ignore")

        if isinstance(self.model, xgb.XGBModel):  # pragma: no cover
            return "Using MSFT interpret is currently unsupported with XGBoost."

        dashboard = []

        if method == "all":
            for explainer in INTERPRET_EXPLAINERS["problem"][self.interpret.problem]:
                dashboard.append(
                    self.interpret.blackbox_show_performance(
                        method=explainer,
                        predictions=predictions,
                        show=False,
                        **interpret_kwargs,
                    )
                )

            if show:
                interpret.show(dashboard)
        else:
            self.interpret.blackbox_show_performance(
                method=method, predictions=predictions, show=show, **interpret_kwargs
            )

    def interpret_model_predictions(
        self,
        num_samples=0.25,
        sample_no=None,
        method="all",
        predictions="default",
        show=True,
        **interpret_kwargs,
    ):
        """
        Plots an interpretable display that explains individual predictions of your model.

        Supported explainers are either 'lime' or 'shap'.

        If 'all' a dashboard is displayed with morris and dependence analysis displayed.
        
        Parameters
        ----------
        num_samples : int, float, or 'all', optional
            Number of samples to display, if less than 1 it will treat it as a percentage, 'all' will include all samples
            , by default 0.25

        sample_no : int, optional
            Sample number to isolate and analyze, if provided it overrides num_samples, by default None

        method : str, optional
            Explainer type, can either be 'all', 'lime', or 'shap', by default 'all'

        predictions : str, optional
            Prediction type, can either be 'default' (.predict) or 'probability' if the model can predict probabilities, by default 'default'

        show : bool, optional 
            False to not display the plot, by default True

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.interpret_model_predictions()
        """

        warnings.simplefilter("ignore")

        if isinstance(self.model, xgb.XGBModel):  # pragma: no cover
            return "Using MSFT interpret is currently unsupported with XGBoost."

        dashboard = []

        if method == "all":
            for explainer in INTERPRET_EXPLAINERS["local"]:
                dashboard.append(
                    self.interpret.blackbox_local_explanation(
                        num_samples=num_samples,
                        sample_no=sample_no,
                        method=explainer,
                        predictions=predictions,
                        show=False,
                        **interpret_kwargs,
                    )
                )

            if show:
                interpret.show(dashboard)
        else:
            self.interpret.blackbox_local_explanation(
                num_samples=num_samples,
                sample_no=sample_no,
                method=method,
                predictions=predictions,
                show=show,
                **interpret_kwargs,
            )

    def interpret_model_behavior(
        self, method="all", predictions="default", show=True, **interpret_kwargs
    ):
        """
        Provides an interpretable summary of your models behaviour based off an explainer.

        Can either be 'morris' or 'dependence' for Partial Dependence.
        
        If 'all' a dashboard is displayed with morris and dependence analysis displayed.
        
        Parameters
        ----------
        method : str, optional
            Explainer type, can either be 'all', 'morris' or 'dependence', by default 'all'

        predictions : str, optional
            Prediction type, can either be 'default' (.predict) or 'probability' if the model can predict probabilities, by default 'default'

        show : bool, optional 
            False to not display the plot, by default True

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.interpret_model_behavior()
        """

        warnings.simplefilter("ignore")

        if isinstance(self.model, xgb.XGBModel):  # pragma: no cover
            return "Using MSFT interpret is currently unsupported with XGBoost."

        dashboard = []

        if method == "all":
            for explainer in INTERPRET_EXPLAINERS["global"]:
                dashboard.append(
                    self.interpret.blackbox_global_explanation(
                        method=explainer,
                        predictions=predictions,
                        show=False,
                        **interpret_kwargs,
                    )
                )

            if show:
                interpret.show(dashboard)
        else:
            self.interpret.blackbox_global_explanation(
                method=method, predictions=predictions, show=show, **interpret_kwargs
            )

    def view_tree(self, tree_num=0, output_file=None, **kwargs):
        """
        Plot decision trees.
        
        Parameters
        ----------
        tree_num: int, optional
            For ensemble, boosting, and stacking methods - the tree number to plot, by default 0

        output_file : str, optional
            Name of the file including extension, by default None

        Examples
        --------
        >>> m = model.DecisionTreeClassifier()
        >>> m.view_tree()
        >>> m = model.XGBoostClassifier()
        >>> m.view_tree(2)
        """

        if hasattr(self, "classes"):
            classes = self.classes
        else:
            classes = None

        if isinstance(self.model, sklearn.tree.BaseDecisionTree):
            graph = Source(
                sklearn.tree.export_graphviz(
                    self.model,
                    out_file=None,
                    feature_names=self.features,
                    class_names=classes,
                    rounded=True,
                    precision=True,
                    filled=True,
                )
            )

            display(SVG(graph.pipe(format="svg")))

        elif isinstance(self.model, xgb.XGBModel):
            return xgb.plot_tree(self.model)

        elif isinstance(self.model, lgb.sklearn.LGBMModel):
            return lgb.plot_tree(self.model)

        elif isinstance(self.model, cb.CatBoost):
            return self.model.plot_tree(tree_idx=tree_num, pool=self.pool)

        elif isinstance(self.model, sklearn.ensemble.BaseEnsemble):
            estimator = self.model.estimators_[tree_num]

            graph = Source(
                sklearn.tree.export_graphviz(
                    estimator,
                    out_file=None,
                    feature_names=self.features,
                    class_names=classes,
                    rounded=True,
                    precision=True,
                    filled=True,
                )
            )

            display(SVG(graph.pipe(format="svg")))
        else:
            raise NotImplementedError(
                f"Model {str(self.model)} cannot be viewed as a tree"
            )

    def to_pickle(self):
        """
        Writes model to a pickle file.

        Examples
        --------
        >>> m = Model(df)
        >>> m_results = m.LogisticRegression()
        >>> m_results.to_pickle()
        """

        to_pickle(self.model, self.model_name)

    def to_service(self, project_name: str):
        """
        Creates an app.py, requirements.txt and Dockerfile in `~/.aethos/projects` and the necessary folder structure
        to run the model as a microservice.
        
        Parameters
        ----------
        project_name : str
            Name of the project that you want to create.

        Examples
        --------
        >>> m = Model(df)
        >>> m_results = m.LogisticRegression()
        >>> m_results.to_service('your_proj_name')
        """

        to_pickle(self.model, self.model_name, project=True, project_name=project_name)
        tg.generate_service(project_name, f"{self.model_name}.pkl")

        print("docker build -t `image_name` ./")
        print("docker run -d --name `container_name` -p `port_num`:80 `image_name`")


class TextModel(ModelBase):
    def __init__(self, model_object, model, model_name, **kwargs):

        super().__init__(model_object, model, model_name)

        self.result_data = model_object.x_train_results

        # LDA dependant variables
        self.corpus = kwargs.pop("corpus", None)
        self.id2word = kwargs.pop("id2word", None)

    def view(self, original_text, model_output):
        """
        View the original text and the model output in a more user friendly format
        
        Parameters
        ----------
        original_text : str
            Column name of the original text

        model_output : str
            Column name of the model text

        Examples
        --------
        >>> m = model.LDA()
        >>> m.view('original_text_col_name', 'model_output_col_name')
        """

        results = self.result_data[[original_text, model_output]]

        display(HTML(results.to_html()))

    def view_topics(self, num_topics=10, **kwargs):
        """
        View topics from topic modelling model.
        
        Parameters
        ----------
        num_topics : int, optional
            Number of topics to view, by default 10

        Returns
        --------
        str
            String representation of topics and probabilities

        Examples
        --------
        >>> m = model.LDA()
        >>> m.view_topics()
        """

        return self.model.show_topics(num_topics=num_topics, **kwargs)

    def view_topic(self, topic_num: int, **kwargs):
        """
        View a specific topic from topic modelling model.
        
        Parameters
        ----------
        topic_num : int

        Returns
        --------
        str
            String representation of topic and probabilities

        Examples
        --------
        >>> m = model.LDA()
        >>> m.view_topic(1)
        """

        return self.model.show_topic(topicid=topic_num, **kwargs)

    def visualize_topics(self, **kwargs):  # pragma: no cover
        """
        Visualize topics using pyLDAvis.

        Parameters
        ----------
        R : int
            The number of terms to display in the barcharts of the visualization. Default is 30. Recommended to be roughly between 10 and 50.

        lambda_step : float, between 0 and 1
            Determines the interstep distance in the grid of lambda values over which to iterate when computing relevance. Default is 0.01. Recommended to be between 0.01 and 0.1.

        mds : function or {'tsne', 'mmds}
            A function that takes topic_term_dists as an input and outputs a n_topics by 2 distance matrix.
            The output approximates the distance between topics. See js_PCoA() for details on the default function.
            A string representation currently accepts pcoa (or upper case variant), mmds (or upper case variant) and tsne (or upper case variant), if sklearn package is installed for the latter two.

        n_jobs : int
            The number of cores to be used to do the computations. The regular joblib conventions are followed so -1, which is the default, will use all cores.

        plot_opts : dict, with keys ‘xlab’ and ylab
            Dictionary of plotting options, right now only used for the axis labels.

        sort_topics : bool
            Sort topics by topic proportion (percentage of tokens covered).
            Set to false to keep original topic order.

        Examples
        --------
        >>> m = model.LDA()
        >>> m.visualize_topics()
        """

        import pyLDAvis
        import pyLDAvis.gensim

        pyLDAvis.enable_notebook()

        return pyLDAvis.gensim.prepare(self.model, self.corpus, self.id2word, **kwargs)

    def coherence_score(self, col_name):
        """
        Displays the coherence score of the topic model.

        For more info on topic coherence: https://rare-technologies.com/what-is-topic-coherence/ 
        
        Parameters
        ----------
        col_name : str
            Column name that was used as input for the LDA model

        Examples
        --------
        >>> m = model.LDA()
        >>> m.coherence_score()
        """

        import gensim

        texts = self.result_data[col_name].tolist()

        coherence_model_lda = gensim.models.CoherenceModel(
            model=self.model, texts=texts, dictionary=self.id2word, coherence="c_v"
        )
        coherence_lda = coherence_model_lda.get_coherence()

        fig = go.Figure(
            go.Indicator(
                domain={"x": [0, 1], "y": [0, 1]},
                value=coherence_lda,
                mode="number",
                title={"text": "Coherence Score"},
            )
        )

        fig.show()

    def model_perplexity(self):
        """
        Displays the model perplexity of the topic model.

        Perplexity is a measurement of how well a probability distribution or probability model predicts a sample. It may be used to compare probability models.
        
        A low perplexity indicates the probability distribution is good at predicting the sample.

        Examples
        --------
        >>> m = model.LDA()
        >>> m.model_perplexity()
        """

        fig = go.Figure(
            go.Indicator(
                domain={"x": [0, 1], "y": [0, 1]},
                value=self.model.log_perplexity(self.corpus),
                mode="number",
                title={"text": "Model Perplexity"},
            )
        )

        fig.show()

        print("Note: The lower the better.")


class UnsupervisedModel(ModelBase):
    def __init__(self, model_object, model_name, model, cluster_col):

        super().__init__(model_object, model, model_name)

        self.cluster_col = cluster_col

        self.x_train[self.cluster_col] = model_object.x_train_results[self.cluster_col]

        if self.x_test is not None:
            self.x_test[self.cluster_col] = model_object.x_test_results[
                self.cluster_col
            ]

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

        if self.x_test is None:
            return self.x_train[self.x_train[self.cluster_col] == cluster_no]
        else:
            return self.x_test[self.x_test[self.cluster_col] == cluster_no]

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

        reduce : str, optional
            Dimension reduction strategy i.e. pca, by default "pca"

        output_file: str
            Output file name including extension (.png, .jpg, etc.) to save image as.

        Examples
        --------
        >>> m = model.KMeans()
        >>> m.plot_clusters()
        >>> m.plot_clusters(dim=3)
        """

        if dim != 2 and dim != 3:
            raise ValueError("Dimension must be either 2d (2) or 3d (3)")

        dataset = self.x_test if self.x_test is not None else self.x_train

        if reduce == "pca":
            reduced_df, _ = pca(
                dataset.drop(self.cluster_col, axis=1),
                n_components=dim,
                random_state=42,
            )
        else:
            raise ValueError("Currently supported dimensionality reducers are: PCA.")

        reduced_df[self.cluster_col] = dataset[self.cluster_col]
        reduced_df.columns = list(map(str, reduced_df.columns))

        if dim == 2:
            scatterplot(
                "0",
                "1",
                data=reduced_df,
                color=reduced_df[self.cluster_col].tolist(),
                output_file=output_file,
                **kwargs,
            )
        else:
            scatterplot(
                "0",
                "1",
                "2",
                data=reduced_df,
                color=self.cluster_col,
                output_file=output_file,
                **kwargs,
            )


class ClassificationModel(ModelBase):
    def __init__(
        self,
        model_object,
        model_name,
        model,
        predictions_col,
        shap_values=None,
        pool=None,
    ):

        self.y_train = model_object.y_train
        self.y_test = (
            model_object.y_test
            if model_object.x_test is not None
            else model_object.y_train
        )

        super().__init__(model_object, model, model_name, shap_values=shap_values)

        self.probabilities = None
        self.target_mapping = model_object.target_mapping
        self.multiclass = len(np.unique(list(self.y_train) + list(self.y_test))) > 2

        self.y_pred = (
            model_object.x_train_results[predictions_col]
            if self.x_test is None
            else model_object.x_test_results[predictions_col]
        )

        if self.report:
            self.report.write_header(f"Analyzing Model {self.model_name.upper()}: ")

        if self.target_mapping is None:
            self.classes = [
                str(item) for item in np.unique(list(self.y_train) + list(self.y_test))
            ]
        else:
            self.classes = [str(item) for item in self.target_mapping.values()]

        self.features = self.x_test.columns
        self.pool = pool

        if hasattr(model, "predict_proba"):
            self.probabilities = model.predict_proba(model_object.x_test)

    def accuracy(self, **kwargs):
        """
        It measures how many observations, both positive and negative, were correctly classified.
        
        Returns
        -------
        float
            Accuracy

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.accuracy()
        """

        return sklearn.metrics.accuracy_score(self.y_test, self.y_pred, **kwargs)

    def balanced_accuracy(self, **kwargs):
        """
        The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.

        The best value is 1 and the worst value is 0 when adjusted=False.
        
        Returns
        -------
        float
            Balanced accuracy

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.balanced_accuracy()
        """

        return sklearn.metrics.balanced_accuracy_score(
            self.y_test, self.y_pred, **kwargs
        )

    def average_precision(self, **kwargs):
        """
        AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold,
        with the increase in recall from the previous threshold used as the weight
        
        Returns
        -------
        float
            Average Precision Score

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.average_precision()
        """

        if hasattr(self.model, "decision_function"):
            return sklearn.metrics.average_precision_score(
                self.y_test, self.model.decision_function(self.x_test), **kwargs
            )
        else:
            return np.nan

    def roc_auc(self, **kwargs):
        """
        This metric tells us that this metric shows how good at ranking predictions your model is.
        It tells you what is the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.
        
        Returns
        -------
        float
            ROC AUC Score

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.roc_auc()
        """

        multi_class = kwargs.pop("multi_class", "ovr")

        if self.multiclass:
            roc_auc = sklearn.metrics.roc_auc_score(
                self.y_test, self.probabilities, multi_class=multi_class, **kwargs
            )
        else:
            roc_auc = sklearn.metrics.roc_auc_score(self.y_test, self.y_pred, **kwargs)

        return roc_auc

    def zero_one_loss(self, **kwargs):
        """
        Return the fraction of misclassifications (float), else it returns the number of misclassifications (int).
        
        The best performance is 0.
        
        Returns
        -------
        float
            Zero one loss

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.zero_one_loss()
        """

        return sklearn.metrics.zero_one_loss(self.y_test, self.y_test, **kwargs)

    def recall(self, **kwargs):
        """
        The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
        
        The recall is intuitively the ability of the classifier to find all the positive samples.

        The best value is 1 and the worst value is 0.
        
        Returns
        -------
        float
            Recall

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.recall()
        """

        avg = kwargs.pop("average", "macro")

        if self.multiclass:
            return sklearn.metrics.recall_score(
                self.y_test, self.y_pred, average=avg, **kwargs
            )
        else:
            return sklearn.metrics.recall_score(self.y_test, self.y_pred, **kwargs)

    def precision(self, **kwargs):
        """
        The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
        
        The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

        The best value is 1 and the worst value is 0.
        
        Returns
        -------
        float
            Precision

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.precision()
        """

        avg = kwargs.pop("average", "macro")

        if self.multiclass:
            return sklearn.metrics.precision_score(
                self.y_test, self.y_pred, average=avg, **kwargs
            )
        else:
            return sklearn.metrics.precision_score(self.y_test, self.y_pred, **kwargs)

    def matthews_corr_coef(self, **kwargs):
        """
        The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications.
        It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
        The MCC is in essence a correlation coefficient value between -1 and +1. 
        A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.
        The statistic is also known as the phi coefficient. 
        
        Returns
        -------
        float
            Matthews Correlation Coefficient

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.mathews_corr_coef()
        """

        return sklearn.metrics.matthews_corrcoef(self.y_test, self.y_pred, **kwargs)

    def log_loss(self, **kwargs):
        """
        Log loss, aka logistic loss or cross-entropy loss.

        This is the loss function used in (multinomial) logistic regression and extensions of it
        such as neural networks, defined as the negative log-likelihood of the true labels given a probabilistic classifier’s predictions.
        
        Returns
        -------
        Float
            Log loss

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.log_loss()
        """

        if self.probabilities is not None:
            return sklearn.metrics.log_loss(self.y_test, self.probabilities, **kwargs)
        else:
            return np.nan

    def jaccard(self, **kwargs):
        """
        The Jaccard index, or Jaccard similarity coefficient,
        defined as the size of the intersection divided by the size of the union of two label sets,
        is used to compare set of predicted labels for a sample to the corresponding set of labels in y_true.
        
        Returns
        -------
        float
            Jaccard Score

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.jaccard()
        """

        avg = kwargs.pop("average", "macro")

        if self.multiclass:
            return sklearn.metrics.jaccard_score(
                self.y_test, self.y_pred, average=avg, **kwargs
            )
        else:
            return sklearn.metrics.jaccard_score(self.y_test, self.y_pred, **kwargs)

    def hinge_loss(self, **kwargs):
        """
        Computes the average distance between the model and the data using hinge loss, a one-sided metric that considers only prediction errors.
        
        Returns
        -------
        float
            Hinge loss

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.hinge_loss()
        """

        if hasattr(self.model, "decision_function"):
            return sklearn.metrics.hinge_loss(
                self.y_test, self.model.decision_function(self.x_test), **kwargs
            )
        else:
            return np.nan

    def hamming_loss(self, **kwargs):
        """
        The Hamming loss is the fraction of labels that are incorrectly predicted.
        
        Returns
        -------
        float
            Hamming loss

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.hamming_loss()
        """

        return sklearn.metrics.hamming_loss(self.y_test, self.y_pred, **kwargs)

    def fbeta(self, beta=0.5, **kwargs):
        """
        The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.
        The beta parameter determines the weight of recall in the combined score.
        Beta < 1 lends more weight to precision, while beta > 1 favors recall (beta -> 0 considers only precision, beta -> inf only recall).
        
        Parameters
        ----------
        beta : float, optional
            Weight of precision in harmonic mean, by default 0.5
        
        Returns
        -------
        float
            Fbeta score

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.fbeta()
        """

        avg = kwargs.pop("average", "macro")

        if self.multiclass:
            return sklearn.metrics.fbeta_score(
                self.y_test, self.y_pred, beta, average=avg, **kwargs
            )
        else:
            return sklearn.metrics.fbeta_score(self.y_test, self.y_pred, beta, **kwargs)

    def f1(self, **kwargs):
        """
        The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:

        F1 = 2 * (precision * recall) / (precision + recall)

        In the multi-class and multi-label case, this is the average of the F1 score of each class with weighting depending on the average parameter.
        
        Returns
        -------
        float
            F1 Score

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.f1()
        """
        avg = kwargs.pop("average", "macro")

        if self.multiclass:
            return sklearn.metrics.f1_score(
                self.y_test, self.y_pred, average=avg, **kwargs
            )
        else:
            return sklearn.metrics.f1_score(self.y_test, self.y_pred, **kwargs)

    def cohen_kappa(self, **kwargs):
        """
        Cohen Kappa tells you how much better is your model over the random classifier that predicts based on class frequencies
        
        This measure is intended to compare labelings by different human annotators, not a classifier versus a ground truth.

        The kappa score (see docstring) is a number between -1 and 1.
        Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels).
        
        Returns
        -------
        float
            Cohen Kappa score.

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.cohen_kappa()
        """

        return sklearn.metrics.cohen_kappa_score(self.y_test, self.y_pred, **kwargs)

    def brier_loss(self, **kwargs):
        """
        Compute the Brier score. The smaller the Brier score, the better, hence the naming with “loss”.  
        Across all items in a set N predictions, the Brier score measures the mean squared difference between (1) the predicted probability assigned to the possible outcomes for item i, and (2) the actual outcome.
        Therefore, the lower the Brier score is for a set of predictions, the better the predictions are calibrated.
        
        The Brier score is appropriate for binary and categorical outcomes that can be structured as true or false,
        but is inappropriate for ordinal variables which can take on three or more values (this is because the Brier score assumes that all possible outcomes are equivalently “distant” from one another)

        Returns
        -------
        float
            Brier loss

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.brier_loss()
        """

        if self.multiclass:
            warnings.warn("Brier Loss can only be used for binary classification.")
            return -999

        return sklearn.metrics.brier_score_loss(self.y_test, self.y_pred, **kwargs)

    def metrics(self, *metrics):
        """
        Measures how well your model performed against certain metrics.

        For multiclassification problems, the 'macro' average is used.

        If a project metrics has been specified, it will display those metrics, otherwise it will display the specified metrics or all metrics.

        For more detailed information and parameters please see the following link: https://scikit-learn.org/stable/modules/classes.html#classification-metrics
        
        Supported metrics are:

            'Accuracy': 'Measures how many observations, both positive and negative, were correctly classified.',
            
            'Balanced Accuracy': 'The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.',
            
            'Average Precision': 'Summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold',
            
            'ROC AUC': 'Shows how good at ranking predictions your model is. It tells you what is the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.',
            
            'Zero One Loss': 'Fraction of misclassifications.',
            
            'Precision': 'It measures how many observations predicted as positive are positive. Good to use when False Positives are costly.',
            
            'Recall': 'It measures how many observations out of all positive observations have we classified as positive. Good to use when catching call positive occurences, usually at the cost of false positive.',
            
            'Matthews Correlation Coefficient': 'It’s a correlation between predicted classes and ground truth.',
            
            'Log Loss': 'Difference between ground truth and predicted score for every observation and average those errors over all observations.',
            
            'Jaccard': 'Defined as the size of the intersection divided by the size of the union of two label sets, is used to compare set of predicted labels for a sample to the corresponding set of true labels.',
            
            'Hinge Loss': 'Computes the average distance between the model and the data using hinge loss, a one-sided metric that considers only prediction errors.',
            
            'Hamming Loss': 'The Hamming loss is the fraction of labels that are incorrectly predicted.',
            
            'F-Beta': 'It’s the harmonic mean between precision and recall, with an emphasis on one or the other. Takes into account both metrics, good for imbalanced problems (spam, fraud, etc.).',
            
            'F1': 'It’s the harmonic mean between precision and recall. Takes into account both metrics, good for imbalanced problems (spam, fraud, etc.).',
            
            'Cohen Kappa': 'Cohen Kappa tells you how much better is your model over the random classifier that predicts based on class frequencies. Works well for imbalanced problems.',
            
            'Brier Loss': 'It is a measure of how far your predictions lie from the true values. Basically, it is a mean square error in the probability space.'
        
        Parameters
        ----------
        metrics : str(s), optional
            Specific type of metrics to view

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.metrics()
        >>> m.metrics('F1', 'F-Beta')
        """

        metric_list = {
            "Accuracy": self.accuracy(),
            "Balanced Accuracy": self.balanced_accuracy(),
            "Average Precision": self.average_precision(),
            "ROC AUC": self.roc_auc(),
            "Zero One Loss": self.zero_one_loss(),
            "Precision": self.precision(),
            "Recall": self.recall(),
            "Matthews Correlation Coefficient": self.matthews_corr_coef(),
            "Log Loss": self.log_loss(),
            "Jaccard": self.jaccard(),
            "Hinge Loss": self.hinge_loss(),
            "Hamming Loss": self.hamming_loss(),
            "F-Beta": self.fbeta(),
            "F1": self.f1(),
            "Cohen Kappa": self.cohen_kappa(),
            "Brier Loss": self.brier_loss(),
        }

        metric_table = pd.DataFrame(
            index=metric_list.keys(),
            columns=[self.model_name],
            data=metric_list.values(),
        )
        metric_table["Description"] = [
            CLASS_METRICS_DESC[x] for x in metric_table.index
        ]

        pd.set_option("display.max_colwidth", -1)

        if not metrics and _global_config["project_metrics"]:  # pragma: no cover
            filt_metrics = _global_config["project_metrics"]
        else:
            filt_metrics = list(metrics) if metrics else metric_table.index

        if self.report:
            self.report.write_metrics(metric_table.loc[filt_metrics, :].round(3))

        return metric_table.loc[filt_metrics, :].round(3)

    def confusion_matrix(
        self,
        title=None,
        normalize=False,
        hide_counts=False,
        x_tick_rotation=0,
        figsize=None,
        cmap="Blues",
        title_fontsize="large",
        text_fontsize="medium",
        output_file="",
    ):
        """
        Prints a confusion matrix as a heatmap.
    
        Arguments
        ---------
        title : str
            The text to display at the top of the matrix, by default 'Confusion Matrix'

        normalize : bool
            If False, plot the raw numbers
            If True, plot the proportions,
            by default False

        hide_counts : bool
            If False, display the counts and percentage
            If True, hide display of the counts and percentage
            by default, False

        x_tick_rotation : int
            Degree of rotation to rotate the x ticks
            by default, 0

        figsize : tuple(int, int)
            Size of the figure
            by default, None

        cmap : str   
            The gradient of the values displayed from matplotlib.pyplot.cm
            see http://matplotlib.org/examples/color/colormaps_reference.html
            plt.get_cmap('jet') or plt.cm.Blues
            by default, 'Blues'

        title_fontsize : str
            Size of the title, by default 'large'

        text_fontsize : str
            Size of the text of the rest of the plot, by default 'medium' 

        output_file: str
            Output file name including extension (.png, .jpg, etc.) to save image as.

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.confusion_matrix()      
        >>> m.confusion_matrix(normalize=True)      
        """

        y_true = self.y_test
        y_pred = self.y_pred

        if figsize:
            plt.figure(figsize=figsize)

        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

        if normalize:
            confusion_matrix = (
                confusion_matrix.astype("float")
                / confusion_matrix.sum(axis=1)[:, np.newaxis]
            )

        accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
        mis_class = 1 - accuracy

        if title:
            plt.title(title, fontsize=title_fontsize)
        elif normalize:
            plt.title("Normalized Confusion Matrix", fontsize=title_fontsize)
        else:
            plt.title("Confusion Matrix", fontsize=title_fontsize)

        cm_sum = np.sum(confusion_matrix, axis=1)
        cm_perc = confusion_matrix / cm_sum.astype(float) * 100
        nrows, ncols = confusion_matrix.shape

        if not hide_counts:
            annot = np.zeros_like(confusion_matrix).astype("str")

            for i in range(nrows):
                for j in range(ncols):
                    c = confusion_matrix[i, j]
                    p = cm_perc[i, j]
                    if i == j:
                        s = cm_sum[i]
                        annot[i, j] = "{:.2f}%\n{}/{}".format(float(p), int(c), int(s))
                    elif c == 0:
                        annot[i, j] = ""
                    else:
                        annot[i, j] = "{:.2f}%\n{}".format(p, c)
        else:
            annot = np.zeros_like(confusion_matrix, dtype=str)

        df_cm = pd.DataFrame(confusion_matrix, index=self.classes, columns=self.classes)

        heatmap = sns.heatmap(
            df_cm, annot=annot, square=True, cmap=plt.cm.get_cmap(cmap), fmt=""
        )

        plt.tight_layout()
        plt.ylabel("True label", fontsize=text_fontsize)
        plt.xlabel(
            "Predicted label\naccuracy={:0.4f}; misclassified={:0.4f}".format(
                accuracy, mis_class
            ),
            fontsize=text_fontsize,
        )
        plt.xticks(
            np.arange(len(self.classes)) + 0.5, self.classes, rotation=x_tick_rotation
        )
        plt.show()

        if output_file:  # pragma: no cover
            image_dir = _make_image_dir()
            heatmap.figure.savefig(os.path.join(image_dir, output_file))

            if self.report:
                self.report.write_image(os.path.join(image_dir, output_file))

    def roc_curve(self, title=True, output_file=""):
        """
        Plots an ROC curve and displays the ROC statistics (area under the curve).

        Parameters
        ----------
        figsize : tuple(int, int), optional
            Figure size, by default (600,450)

        title : bool
            Whether to display title, by default True

        output_file : str, optional
            If a name is provided save the plot to an html file, by default ''

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.roc_curve()
        """

        if self.multiclass:
            raise NotImplementedError(
                "ROC Curve not implemented for multiclassification problems yet."
            )
        else:
            roc_auc = self.roc_auc()

            roc_plot = sklearn.metrics.plot_roc_curve(
                self.model, self.x_test, self.y_test
            )
            roc_plot.ax_.set_xlabel("False Positive Rate or (1 - Specifity)")
            roc_plot.ax_.set_ylabel("True Positive Rate or (Sensitivity)")
            if title:
                roc_plot.figure_.suptitle("ROC Curve (area = {:.2f})".format(roc_auc))

        if output_file:  # pragma: no cover
            image_dir = _make_image_dir()
            roc_plot.figure_.savefig(os.path.join(image_dir, output_file))

            if self.report:
                self.report.write_image(os.path.join(image_dir, output_file))

        return roc_plot

    def classification_report(self):
        """
        Prints and logs the classification report.

        The classification report displays and logs the information in this format:

                    precision    recall  f1-score   support

                    1       1.00      0.67      0.80         3
                    2       0.00      0.00      0.00         0
                    3       0.00      0.00      0.00         0

            micro avg       1.00      0.67      0.80         3
            macro avg       0.33      0.22      0.27         3
         weighted avg       1.00      0.67      0.80         3

        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.classification_report()
        """

        classification_report = sklearn.metrics.classification_report(
            self.y_test, self.y_pred, target_names=self.classes, digits=2
        )

        if self.report:
            self.report.report_classification_report(classification_report)

        print(classification_report)


class RegressionModel(ModelBase):
    def __init__(
        self,
        model_object,
        model_name,
        model,
        predictions_col,
        shap_values=None,
        pool=None,
    ):

        self.y_train = model_object.y_train
        self.y_test = (
            model_object.y_test
            if model_object.x_test is not None
            else model_object.y_train
        )

        super().__init__(model_object, model, model_name, shap_values=shap_values)

        self.y_pred = (
            model_object.x_train_results[predictions_col]
            if self.x_test is None
            else model_object.x_test_results[predictions_col]
        )

        if self.report:
            self.report.write_header(f"Analyzing Model {self.model_name.upper()}: ")

        self.features = self.x_test.columns
        self.pool = pool

    def explained_variance(self, multioutput="uniform_average", **kwargs):
        """
        Explained variance regression score function

        Best possible score is 1.0, lower values are worse.
        
        Parameters
        ----------
        multioutput : string in [‘raw_values’, ‘uniform_average’, ‘variance_weighted’] or array-like of shape (n_outputs)
            Defines aggregating of multiple output scores. Array-like value defines weights used to average scores.

            ‘raw_values’ :
                Returns a full set of scores in case of multioutput input.

            ‘uniform_average’ :
                Scores of all outputs are averaged with uniform weight.

            ‘variance_weighted’ :
                Scores of all outputs are averaged, weighted by the variances of each individual output.

            By default 'uniform_average'
        
        Returns
        -------
        float
            Explained Variance

        Examples
        --------
        >>> m = model.LinearRegression()
        >>> m.explained_variance()
        """

        return sklearn.metrics.explained_variance_score(
            self.y_test, self.y_pred, multioutput="uniform_average", **kwargs
        )

    def max_error(self):
        """
        Returns the single most maximum residual error.
        
        Returns
        -------
        float
            Max error

        Examples
        --------
        >>> m = model.LinearRegression()
        >>> m.max_error()
        """

        return sklearn.metrics.max_error(self.y_test, self.y_pred)

    def mean_abs_error(self, **kwargs):
        """
        Mean absolute error.
        
        Returns
        -------
        float
            Mean absolute error.

        Examples
        --------
        >>> m = model.LinearRegression()
        >>> m.mean_abs_error()
        """

        return sklearn.metrics.mean_absolute_error(self.y_test, self.y_pred)

    def mean_sq_error(self, **kwargs):
        """
        Mean squared error.
        
        Returns
        -------
        float
            Mean squared error.

        Examples
        --------
        >>> m = model.LinearRegression()
        >>> m.mean_sq_error()
        """

        return sklearn.metrics.mean_squared_error(self.y_test, self.y_pred)

    def mean_sq_log_error(self, **kwargs):
        """
        Mean squared log error.
        
        Returns
        -------
        float
            Mean squared log error.

        Examples
        --------
        >>> m = model.LinearRegression()
        >>> m.mean_sq_log_error()
        """

        try:
            return sklearn.metrics.mean_squared_log_error(self.y_test, self.y_pred)
        except ValueError as e:
            warnings.warn(
                "Mean Squared Logarithmic Error cannot be used when targets contain negative values."
            )
            return -999

    def median_abs_error(self, **kwargs):
        """
        Median absolute error.
        
        Returns
        -------
        float
            Median absolute error.

        Examples
        --------
        >>> m = model.LinearRegression()
        >>> m.median_abs_error()
        """

        return sklearn.metrics.median_absolute_error(self.y_test, self.y_pred)

    def r2(self, **kwargs):
        """
        R^2 (coefficient of determination) regression score function.

        R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable
        that is explained by an independent variable or variables in a regression model.

        Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
        A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
        
        Returns
        -------
        float
            R2 coefficient.

        Examples
        --------
        >>> m = model.LinearRegression()
        >>> m.r2()
        """

        return sklearn.metrics.r2_score(self.y_test, self.y_pred)

    def smape(self, **kwargs):
        """
        Symmetric mean absolute percentage error.

        It is an accuracy measure based on percentage (or relative) errors.
        
        Returns
        -------
        float
            SMAPE

        Examples
        --------
        >>> m = model.LinearRegression()
        >>> m.smape()
        """

        return (
            1
            / len(self.y_test)
            * np.sum(
                2
                * np.abs(self.y_pred - self.y_test)
                / (np.abs(self.y_test) + np.abs(self.y_pred))
            )
        )

    def root_mean_sq_error(self):
        """
        Root mean squared error.

        Calculated by taking the square root of the Mean Squared Error.

        Returns
        -------
        float
            Root mean squared error.

        Examples
        --------
        >>> m = model.LinearRegression()
        >>> m.root_mean_sq_error()
        """

        return math.sqrt(self.mean_sq_error())

    def metrics(self, *metrics):
        """
        Measures how well your model performed against certain metrics.

        If a project metrics has been specified, it will display those metrics, otherwise it will display the specified metrics or all metrics.

        For more detailed information and parameters please see the following link: https://scikit-learn.org/stable/modules/classes.html#regression-metrics
        
        Supported metrics are:
            'Explained Variance': 'Explained variance regression score function. Best possible score is 1.0, lower values are worse.',
            
            'Max Error': 'Returns the single most maximum residual error.',
            
            'Mean Absolute Error': 'Postive mean value of all residuals',
            
            'Mean Squared Error': 'Mean of the squared sum the residuals',
            
            'Root Mean Sqaured Error': 'Square root of the Mean Squared Error',
            
            'Mean Squared Log Error': 'Mean of the squared sum of the log of all residuals',
            
            'Median Absolute Error': 'Postive median value of all residuals',
            
            'R2': 'R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable that is explained by an independent variable or variables in a regression model.',
            
            'SMAPE': 'Symmetric mean absolute percentage error. It is an accuracy measure based on percentage (or relative) errors.'

        Parameters
        ----------
        metrics : str(s), optional
            Specific type of metrics to view

        Examples
        --------
        >>> m = model.LinearRegression()
        >>> m.metrics()
        >>> m.metrics('SMAPE', 'Root Mean Squared Error')
        """

        metric_list = {
            "Explained Variance": self.explained_variance(),
            "Max Error": self.max_error(),
            "Mean Absolute Error": self.mean_abs_error(),
            "Mean Squared Error": self.mean_sq_error(),
            "Root Mean Sqaured Error": self.root_mean_sq_error(),
            "Mean Squared Log Error": self.mean_sq_log_error(),
            "Median Absolute Error": self.median_abs_error(),
            "R2": self.r2(),
            "SMAPE": self.smape(),
        }

        metric_table = pd.DataFrame(
            index=metric_list.keys(),
            columns=[self.model_name],
            data=metric_list.values(),
        )
        metric_table["Description"] = [REG_METRICS_DESC[x] for x in metric_table.index]

        pd.set_option("display.max_colwidth", -1)

        if not metrics and _global_config["project_metrics"]:  # pragma: no cover
            filt_metrics = _global_config["project_metrics"]
        else:
            filt_metrics = list(metrics) if metrics else metric_table.index

        if self.report:
            self.report.write_metrics(metric_table.loc[filt_metrics, :].round(3))

        return metric_table.loc[filt_metrics, :].round(3)
