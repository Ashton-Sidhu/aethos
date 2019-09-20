import shap


class Shap(object):

    def __init__(self, model, train_data, test_data, learner: str):

        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        
        if learner == 'linear':
            self.explainer = shap.LinearExplainer(self.model, self.test_data)
        elif learner == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        else:
            raise ValueError('Learner: {} is not supported yet.'.format(learner))
        
        self.expected_value = explainer.expected_value
        self.shap_values = self.explainer.shap_values(self.test_data)
        self.shap_interaction_values = self.explainer.shap_interaction_values(self.test_data)

        # As per SHAP guidelines, test data needs to be dense for plotting functions
        self.test_data_array = self.test_data.toarray()

    def summary_plot(self, **summaryplot_kwargs):
        """
        Plots a SHAP summary plot.
        """

        shap.summary_plot(self.shap_values, self.test_data_array, feature_names=self.train_data.columns, **summaryplot_kwargs)

    def decision_plot(self, **decisionplot_kwargs):
        """
        Plots a SHAP decision plot.
        """

        shap.decision_plot(self.expected_value, self.shap_values, self.test_data_array, **decisionplot_kwargs)

    def force_plot(self, **forceplot_kwargs):
        """
        Plots a SHAP force plot.
        """

        shap.force_plot(self.expected_value, self.shap_values, self.test_data_array, **forceplot_kwargs)

    def dependence_plot(self, feature, **dependenceplot_kwargs):
        """
        Plots a SHAP dependence plot.
        """
        shap.dependence_plot(feature, self.shap_values, self.test_data_array, **dependenceplot_kwargs)
