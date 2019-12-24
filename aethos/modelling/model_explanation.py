import os

import catboost as cb
import interpret
import lightgbm as lgb
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import shap
from aethos.modelling.constants import INTERPRET_EXPLAINERS
from aethos.visualizations.util import _make_image_dir


class Shap(object):
    def __init__(self, model, x_train, x_test, y_test, learner: str, shap_values):

        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_test = y_test

        if learner == "linear":
            self.explainer = shap.LinearExplainer(
                self.model, self.x_train, feature_dependence="independent"
            )
        elif learner == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif learner == "kernel":
            if hasattr(self.model, "predict_proba"):
                func = self.model.predict_proba
            else:
                func = self.model.predict

            self.explainer = shap.KernelExplainer(func, self.x_train)
        else:
            raise ValueError(f"Learner: {learner} is not supported yet.")

        if shap_values is None:
            self.expected_value = self.explainer.expected_value
            self.shap_values = np.array(self.explainer.shap_values(self.x_test)).astype(
                float
            )
        else:
            self.expected_value = shap_values[0, -1]
            self.shap_values = shap_values[:, :-1]

        if isinstance(self.model, lgb.sklearn.LGBMClassifier) and isinstance(
            self.expected_value, np.float
        ):
            self.shap_values = self.shap_values[1]

        # Calculate misclassified values
        self.misclassified_values = self._calculate_misclassified()

        # As per SHAP guidelines, test data needs to be dense for plotting functions
        self.x_test_array = self.x_test.values

    def summary_plot(self, output_file="", **summaryplot_kwargs):
        """
        Plots a SHAP summary plot.

        Parameters
        ----------
        output_file: str
            Output file name including extension (.png, .jpg, etc.) to save image as.
        """

        shap.summary_plot(
            self.shap_values,
            self.x_test_array,
            feature_names=self.x_train.columns,
            show=False,
            **summaryplot_kwargs,
        )

        if output_file:  # pragma: no cover
            image_dir = _make_image_dir()
            pl.savefig(os.path.join(image_dir, output_file))

    def decision_plot(
        self, num_samples=0.25, sample_no=None, output_file="", **decisionplot_kwargs
    ):
        """
        Plots a SHAP decision plot.
        
        Parameters
        ----------
        num_samples : int, float, or 'all', optional
            Number of samples to display, if less than 1 it will treat it as a percentage, 'all' will include all samples
            , by default 0.25

        sample_no : int, optional
            Sample number to isolate and analyze, if provided it overrides num_samples, by default None

        Returns
        -------
        DecisionPlotResult 
            If return_objects=True (the default). Returns None otherwise.
        """

        return_objects = decisionplot_kwargs.pop("return_objects", True)
        highlight = decisionplot_kwargs.pop("highlight", None)

        if sample_no is not None:
            if sample_no < 1 or not isinstance(sample_no, int):
                raise ValueError("Sample number must be greater than 1.")

            samples = slice(sample_no - 1, sample_no)
        else:
            if num_samples == "all":
                samples = slice(0, len(self.x_test_array))
            elif num_samples <= 0:
                raise ValueError(
                    "Number of samples must be greater than 0. If it is less than 1, it will be treated as a percentage."
                )
            elif num_samples > 0 and num_samples < 1:
                samples = slice(0, int(num_samples * len(self.x_test_array)))
            else:
                samples = slice(0, num_samples)

        if highlight is not None:
            highlight = highlight[samples]

        s = shap.decision_plot(
            self.expected_value,
            self.shap_values[samples],
            self.x_train.columns,
            return_objects=return_objects,
            highlight=highlight,
            show=False,
            **decisionplot_kwargs,
        )

        if output_file:  # pragma: no cover
            image_dir = _make_image_dir()
            pl.savefig(os.path.join(image_dir, output_file))

        return s

    def force_plot(self, sample_no=None, output_file="", **forceplot_kwargs):
        """
        Plots a SHAP force plot.
        """

        shap_values = forceplot_kwargs.pop("shap_values", self.shap_values)

        if sample_no is not None:
            if sample_no < 1 or not isinstance(sample_no, int):
                raise ValueError("Sample number must be greater than 1.")

            samples = slice(sample_no - 1, sample_no)
        else:
            samples = slice(0, len(shap_values))

        s = shap.force_plot(
            self.expected_value,
            shap_values[samples],
            self.x_train.columns,
            **forceplot_kwargs,
        )

        if output_file:  # pragma: no cover
            image_dir = _make_image_dir()
            pl.savefig(os.path.join(image_dir, output_file))

        return s

    def dependence_plot(
        self, feature, interaction=None, output_file="", **dependenceplot_kwargs
    ):
        """
        Plots a SHAP dependence plot.
        """

        interaction = dependenceplot_kwargs.pop("interaction_index", interaction)

        shap.dependence_plot(
            feature,
            self.shap_values,
            self.x_test,
            interaction_index=interaction,
            show=False,
            **dependenceplot_kwargs,
        )

        if output_file:  # pragma: no cover
            image_dir = _make_image_dir()
            pl.savefig(os.path.join(image_dir, output_file))

    def _calculate_misclassified(self) -> list:
        """
        Calculates misclassified points.
        
        Returns
        -------
        list
            List specifying which values were misclassified
        """

        if len(self.shap_values.shape) > 2:
            y_pred = [
                x.sum(1) + y > 0 for x, y in zip(self.shap_values, self.expected_value)
            ]
            misclassified = [x != self.y_test for x in y_pred]
        else:
            y_pred = (self.shap_values.sum(1) + self.expected_value) > 0
            misclassified = y_pred != self.y_test

        return misclassified


class MSFTInterpret(object):
    def __init__(self, model, x_train, x_test, y_test, problem):

        self.model = model
        self.x_train = x_train.apply(pd.to_numeric)
        self.x_test = x_test.apply(pd.to_numeric)
        self.y_test = y_test
        self.problem = problem
        self.trained_blackbox_explainers = {}

    def blackbox_show_performance(self, method, predictions="default", show=True):
        """
        Plots an interpretable display of your model based off a performance metric.

        Can either be 'ROC' or 'PR' for precision, recall for classification problems.

        Can be 'regperf' for regression problems.
        
        Parameters
        ----------
        method : str
            Performance metric, either 'roc' or 'PR'

        predictions : str, optional
            Prediction type, can either be 'default' (.predict) or 'probability' if the model can predict probabilities, by default 'default'

        show : bool, optional
            False to not display the plot, by default True
        
        Returns
        -------
        Interpret
            Interpretable dashboard of your model
        """

        if predictions == "probability":
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict

        if self.problem in INTERPRET_EXPLAINERS["problem"]:
            if method.lower() in INTERPRET_EXPLAINERS["problem"][self.problem]:
                blackbox_perf = INTERPRET_EXPLAINERS["problem"][self.problem][
                    method.lower()
                ](predict_fn).explain_perf(
                    self.x_test, self.y_test, name=method.upper()
                )
        else:
            raise ValueError(
                "Supported blackbox explainers are only {} for classification problems and {} for regression problems".format(
                    ",".join(INTERPRET_EXPLAINERS["problem"]["classification"].keys()),
                    ",".join(INTERPRET_EXPLAINERS["problem"]["regression"].keys()),
                )
            )

        if show:
            interpret.show(blackbox_perf)

        self.trained_blackbox_explainers[method.lower()] = blackbox_perf

        return blackbox_perf

    def blackbox_local_explanation(
        self,
        num_samples=0.5,
        sample_no=None,
        method="lime",
        predictions="default",
        show=True,
        **kwargs,
    ):
        """
        Plots an interpretable display that explains individual predictions of your model.

        Supported explainers are either 'lime' or 'shap'.
        
        Parameters
        ----------
        num_samples : int, float, or 'all', optional
            Number of samples to display, if less than 1 it will treat it as a percentage, 'all' will include all samples
            , by default 0.25

        sample_no : int, optional
            Sample number to isolate and analyze, if provided it overrides num_samples, by default None

        method : str, optional
            Explainer type, can either be 'lime' or 'shap', by default 'lime'

        predictions : str, optional
            Prediction type, can either be 'default' (.predict) or 'probability' if the model can predict probabilities, by default 'default'

        show : bool, optional
            False to not display the plot, by default True
        
        Returns
        -------
        Interpret
            Interpretable dashboard of your model
        """

        if predictions == "probability":
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict

        # Determine method
        if method.lower() in INTERPRET_EXPLAINERS["local"]:
            if method.lower() == "lime":
                data = self.x_train
            elif method.lower() == "shap":
                data = np.median(self.x_train, axis=0).reshape(1, -1)
            else:
                raise ValueError

            explainer = INTERPRET_EXPLAINERS["local"][method.lower()](
                predict_fn=predict_fn, data=data, **kwargs
            )

        else:
            raise ValueError(
                'Supported blackbox local explainers are only "lime" and "shap".'
            )

        if sample_no is not None:
            if sample_no < 1 or not isinstance(sample_no, int):
                raise ValueError("Sample number must be greater than 1.")

            samples = slice(sample_no - 1, sample_no)
        else:
            if num_samples == "all":
                samples = slice(0, len(self.x_test))
            elif num_samples <= 0:
                raise ValueError(
                    "Number of samples must be greater than 0. If it is less than 1, it will be treated as a percentage."
                )
            elif num_samples > 0 and num_samples < 1:
                samples = slice(0, int(num_samples * len(self.x_test)))
            else:
                samples = slice(0, num_samples)

        explainer_local = explainer.explain_local(
            self.x_test[samples], self.y_test[samples], name=method.upper()
        )

        self.trained_blackbox_explainers[method.lower()] = explainer_local

        if show:
            interpret.show(explainer_local)

        return explainer_local

    def blackbox_global_explanation(
        self, method="morris", predictions="default", show=True, **kwargs
    ):
        """
        Provides an interpretable summary of your models behaviour based off an explainer.

        Can either be 'morris' or 'dependence' for Partial Dependence.
        
        Parameters
        ----------
        method : str, optional
            Explainer type, can either be 'morris' or 'dependence', by default 'morris'

        predictions : str, optional
            Prediction type, can either be 'default' (.predict) or 'probability' if the model can predict probabilities, by default 'default'

        show : bool, optional
            False to not display the plot, by default True
        
        Returns
        -------
        Interpret
            Interpretable dashboard of your model
        """

        if predictions == "probability":
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict

        if method.lower() in INTERPRET_EXPLAINERS["global"]:
            sensitivity = INTERPRET_EXPLAINERS["global"][method.lower()](
                predict_fn=predict_fn, data=self.x_train, **kwargs
            )

        else:
            raise ValueError(
                'Supported blackbox global explainers are only "morris" and partial "dependence".'
            )

        sensitivity_global = sensitivity.explain_global(name=method.upper())

        self.trained_blackbox_explainers[method.lower()] = sensitivity_global

        if show:
            interpret.show(sensitivity_global)

        return sensitivity_global

    def create_dashboard(self):  # pragma: no cover
        """
        Displays an interpretable dashboard of already created interpretable plots.
        
        If a plot hasn't been interpreted yet it is created using default parameters for the dashboard.
        """

        dashboard_plots = []

        for explainer_type in INTERPRET_EXPLAINERS:

            if explainer_type == "problem":
                temp_explainer_type = INTERPRET_EXPLAINERS[explainer_type][self.problem]
            else:
                temp_explainer_type = INTERPRET_EXPLAINERS[explainer_type]

            for explainer in temp_explainer_type:
                if explainer in self.trained_blackbox_explainers:
                    dashboard_plots.append(self.trained_blackbox_explainers[explainer])
                else:
                    if explainer_type == "problem":
                        dashboard_plots.append(
                            self.blackbox_show_performance(explainer, show=False)
                        )
                    elif explainer_type == "local":
                        dashboard_plots.append(
                            self.blackbox_local_explanation(
                                method=explainer, show=False
                            )
                        )
                    else:
                        dashboard_plots.append(
                            self.blackbox_global_explanation(
                                method=explainer, show=False
                            )
                        )

        interpret.show(dashboard_plots)
