import itertools
import math
import os
import warnings
from collections import OrderedDict
from itertools import compress

import xgboost as xgb
import numpy as np
import pandas as pd
import sklearn
from IPython.display import HTML, SVG, display

from aethos.config.config import _global_config
from aethos.feature_engineering.util import sklearn_dim_reduction
from aethos.model_analysis.model_explanation import MSFTInterpret, Shap
from aethos.modelling.util import (
    to_pickle,
    track_artifacts,
    _get_cv_type,
    run_crossvalidation,
)
from aethos.templates.template_generator import TemplateGenerator as tg
from aethos.visualizations.visualizations import Visualizations
from aethos.stats.stats import Stats
from aethos.model_analysis.constants import (
    PROBLEM_TYPE,
    SHAP_LEARNERS,
)


class ModelAnalysisBase(Visualizations, Stats):

    # TODO: Add more SHAP use cases

    def _repr_html(self):

        if hasattr(self, "x_test"):
            data = self.test_results
        else:
            data = self.train_results

        return data

    @property
    def train_results(self):

        data = self.x_train.copy()
        data["predicted"] = self.model.predict(data)
        data["actual"] = self.y_train

        return data

    @property
    def test_results(self):

        data = self.x_test
        data["actual"] = self.y_test

        data["predicted"] = self.y_pred

        return data

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
        tg.generate_service(project_name, f"{self.model_name}.pkl", self.model)

        print("To run:")
        print("\tdocker build -t `image_name` ./")
        print("\tdocker run -d --name `container_name` -p `port_num`:80 `image_name`")


class SupervisedModelAnalysis(ModelAnalysisBase):
    def __init__(self, model, x_train, x_test, y_train, y_test, model_name):

        self.model = model
        self.model_name = model_name
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = x_test.columns
        self.y_pred = self.model.predict(
            self.x_test[self.features]
        )  # Specifying columns for XGBoost
        self.run_id = None

        if hasattr(model, "predict_proba"):
            self.probabilities = self.model.predict_proba(self.x_test[self.features])

        self.shap = Shap(
            self.model,
            self.model_name,
            self.x_train,
            self.x_test,
            self.y_test,
            SHAP_LEARNERS[type(self.model)],
        )
        self.interpret = MSFTInterpret(
            self.model,
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            PROBLEM_TYPE[type(self.model)],
        )

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

        if _global_config["track_experiments"]:  # pragma: no cover
            track_artifacts(self.run_id, self.model_name)

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

        if _global_config["track_experiments"]:  # pragma: no cover
            track_artifacts(self.run_id, self.model_name)

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

        if _global_config["track_experiments"]:  # pragma: no cover
            track_artifacts(self.run_id, self.model_name)

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

        if _global_config["track_experiments"]:  # pragma: no cover
            track_artifacts(self.run_id, self.model_name)

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

        return sample_list

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

        import interpret
        from aethos.model_analysis.constants import INTERPRET_EXPLAINERS

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

        import interpret
        from aethos.model_analysis.constants import INTERPRET_EXPLAINERS

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

        import interpret
        from aethos.model_analysis.constants import INTERPRET_EXPLAINERS

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

        import lightgbm as lgb
        from graphviz import Source

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

    def _cross_validate(self, cv_type, score, n_splits, shuffle, **kwargs):
        """Runs crossvalidation on a model"""

        cv = _get_cv_type(cv_type, n_splits, shuffle, **kwargs)

        cv_scores = run_crossvalidation(
            self.model,
            self.x_train,
            self.y_train,
            cv=cv,
            scoring=score,
            model_name=self.model_name,
        )
