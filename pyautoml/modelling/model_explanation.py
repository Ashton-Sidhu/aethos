import numpy as np
import shap


class Shap(object):

    def __init__(self, model, train_data, test_data, y_test, learner: str):

        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.y_test = y_test
        
        if learner == 'linear':
            self.explainer = shap.LinearExplainer(self.model, self.train_data, feature_dependence='independent')
        elif learner == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_interaction_values = self.explainer.shap_interaction_values(self.test_data)
        else:
            raise ValueError('Learner: {} is not supported yet.'.format(learner))
        
        self.expected_value = self.explainer.expected_value
        self.shap_values = np.array(self.explainer.shap_values(self.test_data)).astype(float)
        
        # Calculate misclassified values
        self.misclassified_values = self._calculate_misclassified()

        # As per SHAP guidelines, test data needs to be dense for plotting functions
        self.test_data_array = self.test_data.values

    def summary_plot(self, **summaryplot_kwargs):
        """
        Plots a SHAP summary plot.
        """

        shap.summary_plot(self.shap_values, self.test_data_array, feature_names=self.train_data.columns, **summaryplot_kwargs)

    def decision_plot(self, num_samples=0.6, sample_no=None, **decisionplot_kwargs):
        """
        Plots a SHAP decision plot.
        
        Parameters
        ----------
        num_samples : int, float, or 'all', optional
            Number of samples to display, if less than 1 it will treat it as a percentage, 'all' will include all samples
            , by default 0.6

        sample_no : int, optional
            Sample number to isolate and analyze, if provided it overrides num_samples, by default None

        Returns
        -------
        DecisionPlotResult 
            If return_objects=True (the default). Returns None otherwise.
        """

        return_objects = decisionplot_kwargs.pop('return_objects', True)
        highlight = decisionplot_kwargs.pop('highlight', None)

        if sample_no is not None:
            if sample_no < 1 or not isinstance(sample_no, int):
                raise ValueError('Sample number must be greater than 1.')

            samples = slice(sample_no - 1, sample_no)
        else:
            if num_samples == 'all':
                samples = slice(0, len(self.test_data_array))
            elif num_samples <= 0:
                raise ValueError('Number of samples must be greater than 0. If it is less than 1, it will be treated as a percentage.')
            elif num_samples > 0 and num_samples < 1:
                samples = slice(0, int(num_samples * len(self.test_data_array)))
            else:
                samples = slice(0, num_samples)

        if highlight is not None:
            highlight = highlight[samples]

        return shap.decision_plot(self.expected_value, self.shap_values[samples], self.train_data.columns, return_objects=return_objects, highlight=highlight, **decisionplot_kwargs) 

    def force_plot(self, sample_no=None, **forceplot_kwargs):
        """
        Plots a SHAP force plot.
        """

        shap_values = forceplot_kwargs.pop('shap_values', self.shap_values)

        if sample_no is not None:
            if sample_no < 1 or not isinstance(sample_no, int):
                raise ValueError('Sample number must be greater than 1.')

            samples = slice(sample_no - 1, sample_no)
        else:
            samples = slice(0, len(shap_values))

        return shap.force_plot(self.expected_value, shap_values[samples], self.train_data.columns, **forceplot_kwargs)

    def dependence_plot(self, feature, interaction=None, **dependenceplot_kwargs):
        """
        Plots a SHAP dependence plot.
        """

        interaction = dependenceplot_kwargs.pop('interaction_index', interaction)

        shap.dependence_plot(feature, self.shap_values, self.test_data, interaction_index=interaction, **dependenceplot_kwargs)

    def _calculate_misclassified(self) -> list:
        """
        Calculates misclassified points.
        
        Returns
        -------
        list
            List specifying which values were misclassified
        """

        y_pred = (self.shap_values.sum(1) + self.expected_value) > 0
        misclassified = y_pred != self.y_test

        return misclassified
