import itertools
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from bokeh.io import show
from bokeh.models import BoxSelectTool
from bokeh.plotting import figure, output_file
from pyautoml.visualizations.visualize import *

SCORE_METRICS = [
    'accuracy',
    'average_precision',
    'balanced_accuracy',
    'cohen_kappa',
    'f1',
    'jaccard',
    'precision',
    'recall',
    'roc_auc',
]
class ModelBase(object):

    def __init__(self, model_object, model_name):

        self.model_name = model_name
        self.data = model_object._data_properties.data
        self.train_data = model_object._data_properties.train_data
        self.test_data = model_object._data_properties.test_data
        self.report = model_object._data_properties.report

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
        """

        report_strings = []

        try:
            model_dict = dict(zip(self.features, self.model.coef_.flatten()))
        except Exception as e:
            raise AttributeError('Model does not have coefficients to view.')

        sorted_features = OrderedDict(sorted(model_dict.items(), key=lambda kv: abs(kv[1]), reverse=True))

        for feature, weight in sorted_features.items():
            report_string = '\t{} : {:.2f}'.format(feature, weight)
            report_strings.append(report_string)

            print(report_string.strip())

        if self.report:
            self.report.log('Features ranked from most to least important:\n')
            self.report.write_contents("\n".join(report_strings))

        return sorted_features

class TextModel(ModelBase):

    def __init__(self, model_object, model_name):

        super().__init__(model_object, model_name)

class ClusterModel(ModelBase):

    def __init__(self, model_object, model_name, model, cluster_col):

        super().__init__(model_object, model_name)

        self.model = model
        self.cluster_col = cluster_col

        if self.data is not None:
            self.prediction_data = self.data[cluster_col]
        else:
            self.train_prediction_data = self.train_data[cluster_col]
            self.test_prediction_data = self.test_data[cluster_col]

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
        """

        if self.data is not None:
            return self.data[self.data[self.cluster_col] == cluster_no]
        else:
            return self.test_data[self.test_data[self.cluster_col] == cluster_no]

class ClassificationModel(ModelBase):

    def __init__(self, model_object, model_name, model, predictions_col):

        super().__init__(model_object, model_name)

        self.model = model
        self.target_mapping = model_object.target_mapping

        if self.data is not None:
            self.target_data = model_object.target_data
            self.prediction_data = self.data[predictions_col]
        else:
            self.target_data = model_object.test_target_data
            self.prediction_data = self.test_data[predictions_col]

        if self.report:
            self.report.write_header('Analyzing Model {}: '.format(self.model_name.upper()))

        if self.target_mapping is None:
            self.classes = [str(item) for item in np.unique(list(self.target_data) + list(self.prediction_data))]
        else:
            self.classes = [str(item) for item in self.target_mapping.values()]

        self.features = self.test_data.columns

    def metric(self, *metrics, metric='accuracy', **scoring_kwargs):
        """
        Measures how well your model performed based off a certain metric. It can be any combination of the ones below or 'all' for 
        every metric listed below. The default measure is accuracy.

        For more detailed information and parameters please see the following link: https://scikit-learn.org/stable/modules/classes.html#classification-metrics

        Supported metrics are:

            all : Everything below.
            accuracy : Accuracy classification score.
            average_precision : Compute average precision (AP) from prediction scores
            balanced_accuracy : Compute the balanced accuracy
            cohen_kappa : Cohen’s kappa: a statistic that measures inter-annotator agreement.
            f1 : Compute the F1 score, also known as balanced F-score or F-measure
            fbeta : Compute the F-beta score
            jaccard : Jaccard similarity coefficient score
            precision : Compute the precision
            recall : Compute the recall
            roc_auc : Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        
        Parameters
        ----------
        metric : str, optional
            Specific type of metric, by default 'accuracy'
        """

        y_true = self.target_data
        y_pred = self.prediction_data
        computed_metrics = []


        if metric == 'all' or 'all' in metrics:
            for met in SCORE_METRICS:
                metric_str = '{} : {}'.format(met, getattr(sklearn.metrics, met + "_score")(y_true, y_pred))
                computed_metrics.append(metric_str)
                print(metric_str)
        elif metrics:
            for met in metrics:
                metric_str = '{} : {}'.format(met, getattr(sklearn.metrics, met + "_score")(y_true, y_pred))
                computed_metrics.append(metric_str)
                print(metric_str)
        else:      
            metric_str = '{} : {}'.format(met, getattr(sklearn.metrics, met + "_score")(y_true, y_pred, **scoring_kwargs))
            computed_metrics.append(metric_str)
            print(metric_str)

        if self.report:
            self.report.log('Metrics:\n')
            self.report.log('\n'.join(computed_metrics))

    def confusion_matrix(self, title=None, normalize=False, hide_counts=False, x_tick_rotation=0, figsize=None, cmap='Blues', title_fontsize="large", text_fontsize="medium"):
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
        """
        
        y_true = self.target_data
        y_pred = self.prediction_data

        if figsize:
            plt.figure(figsize=figsize)

        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

        accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
        mis_class = 1 - accuracy

        if title:
            plt.title(title, fontsize=title_fontsize)
        elif normalize:
            plt.title('Normalized Confusion Matrix', fontsize=title_fontsize)
        else:
            plt.title('Confusion Matrix', fontsize=title_fontsize)

        cm_sum = np.sum(confusion_matrix, axis=1)
        cm_perc = confusion_matrix / cm_sum.astype(float) * 100
        nrows, ncols = confusion_matrix.shape

        if not hide_counts:
            annot = np.zeros_like(confusion_matrix).astype('str')

            for i in range(nrows):
                for j in range(ncols):
                    c = confusion_matrix[i, j]
                    p = cm_perc[i, j]
                    if i == j:
                        s = cm_sum[i]
                        annot[i, j] = '{:.2f}%\n{}/{}'.format(float(p), int(c), int(s))
                    elif c == 0:
                        annot[i, j] = ''
                    else:
                        annot[i, j] = '{:.2f}%\n{}'.format(p, c)
        else:
            annot = np.zeros_like(confusion_matrix, dtype=str)

        df_cm = pd.DataFrame(
            confusion_matrix, index=self.classes, columns=self.classes, 
        )

        heatmap = sns.heatmap(df_cm, annot=annot, square=True, cmap=plt.cm.get_cmap(cmap), fmt='')       

        plt.tight_layout()
        plt.ylabel('True label', fontsize=text_fontsize)
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclassified={:0.4f}'.format(accuracy, mis_class), fontsize=text_fontsize)
        plt.xticks(np.arange(len(self.classes)) + 0.5, self.classes, rotation=x_tick_rotation)
        plt.show()

        if self.report:
            self.report.log('CONFUSION MATRIX:\n')
            self.report.log(df_cm.to_string())

    def roc_curve(self, figsize=(450,550), output_file=''):
        """
        Plots an ROC curve and displays the ROC statistics (area under the curve).

        Parameters
        ----------
        figsize : tuple(int, int), optional
            Figure size, by default (450,550)

        output_file : str, optional
            If a name is provided save the plot to an html file, by default ''
        """

        if len(np.unique(list(self.target_data) + list(self.prediction_data))) > 2:
            raise NotImplementedError('ROC Curve is currently not implemented for multiclassification problems.')

        y_true = self.target_data
        y_pred = self.prediction_data

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
        roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred)

        step = 1 / (len(fpr) - 1)
        random = np.arange(0, 1 + step, step)

        p = figure(plot_width=figsize[0], plot_height=figsize[1], title='ROC Curve (Area = {:.2f})'.format(roc_auc), x_range=[0,1], y_range=[0,1], x_axis_label='False Positive Rate or (1 - Specifity)', y_axis_label='True Positive Rate or (Sensitivity)', tooltips=[('False Positive Rate', '$x'), ('True Positve Rate', '$y')], tools='pan,wheel_zoom,tap,box_zoom,reset', active_drag='box_zoom', active_scroll='wheel_zoom')

        p.line(fpr, tpr, color='blue', alpha=0.8, legend='ROC')
        p.line(random, random, color='orange', line_dash='dashed', legend='Baseline')

        p.legend.location = "bottom_right"
        p.legend.click_policy = "hide"

        if output_file:
            output_file(output_file + '.html', title='ROC Curve (area = {:.2f})'.format(roc_auc))

        show(p)

    # TODO: MSFT Interpret
    # TODO: SHAP
    
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
        """

        classification_report = sklearn.metrics.classification_report(self.target_data, self.target_data, target_names=self.classes, digits=2)

        if self.report:
            self.report.report_classification_report(classification_report)

        print(classification_report)        

class RegressionModel(ModelBase):
    # TODO: Summary statistics
    # TODO: Errors
    # TODO: MSFT Interpret
    # TODO: SHAP

    pass
