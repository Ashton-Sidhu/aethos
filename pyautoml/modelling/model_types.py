import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn


class ModelBase(object):

    def __init__(self, model_object):

        self.data = model_object._data_properties.data
        self.train_data = model_object._data_properties.train_data
        self.test_data = model_object._data_properties.test_data

class TextModel(ModelBase):

    def __init__(self, model_object):

        super().__init__(model_object)

class ClusterModel(ModelBase):

    def __init__(self, model_object, model, cluster_col):

        super().__init__(model_object)

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

    def __init__(self, model_object, model, predictions_col):

        super().__init__(model_object)

        self.model = model
        self.target_mapping = model_object.target_mapping

        if self.data is not None:
            self.target_data = model_object.target_data
            self.prediction_data = self.data[predictions_col]
        else:
            self.target_data = model_object.test_target_data
            self.prediction_data = self.test_data[predictions_col]

    def confusion_matrix(self, title='Confusion Matrix', cmap=None, normalize=False):
        """
        Plots a confusion matrix based off your test data.

        Parameters
        ----------
        title:        
            The text to display at the top of the matrix

        cmap: str
            The gradient of the values displayed from matplotlib.pyplot.cm
            see http://matplotlib.org/examples/color/colormaps_reference.html
            plt.get_cmap('jet') or plt.cm.Blues

        normalize: str   
            If False, plot the raw numbers
            If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(
                            normalize    = True,                # show proportions
                            target_names = y_labels_vals,       # list of names of the classes
                            title        = best_estimator_name) # title of graph

        Citiation
        ---------
        https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

        """
        cm = sklearn.metrics.confusion_matrix(self.target_data, self.prediction_data)

        accuracy = np.trace(cm) / float(np.sum(cm))
        mis_class = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")


        if self.target_mapping is not None:
            tick_marks = np.arange(len(self.target_mapping))
            plt.xticks(tick_marks, self.target_mapping.keys(), rotation=45)
            plt.yticks(tick_marks, self.target_mapping.keys())
        else:
            tick_marks = np.unique(self.target_data)
            plt.xticks(tick_marks)
            plt.ylim(tick_marks)
            plt.yticks(tick_marks)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclassified={:0.4f}'.format(accuracy, mis_class))
        plt.show()

    def plot_cmat(self, title=None, normalize=False,
                    hide_zeros=False, hide_counts=False, x_tick_rotation=0,
                    figsize=(10,7), cmap='Blues', title_fontsize="large",
                    text_fontsize="medium"):
        """
        Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
        Arguments
        ---------
        confusion_matrix: numpy.ndarray
            The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
            Similarly constructed ndarrays can also be used.
        class_names: list
            An ordered list of class names, in the order they index the given confusion matrix.
        figsize: tuple
            A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
            the second determining the vertical size. Defaults to (10,7).
        fontsize: int
            Font size for axes labels. Defaults to 14.
            
        Returns
        -------
        matplotlib.figure.Figure
            The resulting confusion matrix figure
        """
        
        y_true = self.target_data
        y_pred = self.prediction_data
        fmt = 'd' if normalize else '.2f'

        if self.target_mapping is None:
            classes = list(map(str, np.unique(list(y_true) + list(y_pred))))
        else:
            classes = list(map(str, self.target_mapping.values()))

        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

        accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
        mis_class = 1 - accuracy

        df_cm = pd.DataFrame(
            confusion_matrix, index=classes, columns=classes, 
        )

        if title:
            plt.title(title, fontsize=title_fontsize)
        elif normalize:
            plt.title('Normalized Confusion Matrix', fontsize=title_fontsize)
        else:
            plt.title('Confusion Matrix', fontsize=title_fontsize)

        annot = True if not hide_counts else False        

        try:
            heatmap = sns.heatmap(df_cm, annot=annot, fmt=fmt, cmap=plt.cm.get_cmap(cmap))
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")        

        # plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=0, va="center", fontsize=text_fontsize)
        # plt.xticks(np.arange(len(classes)) + 0.5, classes, rotation=45, fontsize=text_fontsize)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclassified={:0.4f}'.format(accuracy, mis_class))

        plt.show()

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
