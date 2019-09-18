import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

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
            self.report.write_header('Analyzing Model {}'.format(self.model_name))

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

        try:
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
        except Exception as e:
            print('Could not calculate metric, due to {}').format(e)

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

        if self.target_mapping is None:
            classes = np.unique(list(y_true) + list(y_pred))
        else:
            classes = self.target_mapping.values()

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
            confusion_matrix, index=classes, columns=classes, 
        )

        heatmap = sns.heatmap(df_cm, annot=annot, square=True, cmap=plt.cm.get_cmap(cmap), fmt='')       

        plt.tight_layout()
        plt.ylabel('True label', fontsize=text_fontsize)
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclassified={:0.4f}'.format(accuracy, mis_class), fontsize=text_fontsize)
        plt.xticks(np.array(classes) + 0.5, classes, rotation=x_tick_rotation)
        plt.show()

        if self.report:
            self.report.log('CONFUSION MATRIX:\n')
            self.report.log(df_cm.to_string())

    # TODO: Precision, Recall, F1
    # TODO: ROC Curve
    # TODO: Model Weights
    # TODO: MSFT Interpret
    # TODO: SHAP
    # TODO: classification report

class RegressionModel(ModelBase):
    # TODO: Summary statistics
    # TODO: Errors
    # TODO: Model Weights
    # TODO: MSFT Interpret
    # TODO: SHAP

    pass
