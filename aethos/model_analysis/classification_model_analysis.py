import os
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics

from aethos.config import IMAGE_DIR
from aethos.config.config import _global_config
from .model_analysis import SupervisedModelAnalysis
from aethos.modelling.util import track_artifacts


class ClassificationModelAnalysis(SupervisedModelAnalysis):
    def __init__(
        self, model, x_train, x_test, target, model_name,
    ):
        """
        Class to analyze Classification models through metrics, global/local interpretation and visualizations.

        Parameters
        ----------
        model : str or Model Object
            Sklearn, XGBoost, LightGBM Model object or .pkl file of the objects.

        x_train : pd.DataFrame
            Training Data used for the model.

        x_test : pd.DataFrame
            Test data used for the model.

        target : str
            Target column in the DataFrame

        model_name : str
            Name of the model for saving images and model tracking purposes
        """

        # TODO: Add check for pickle file

        super().__init__(
            model,
            x_train.drop(target, axis=1),
            x_test.drop(target, axis=1),
            x_train[target],
            x_test[target],
            model_name,
        )

        self.multiclass = len(np.unique(list(self.y_train) + list(self.y_test))) > 2

        self.classes = [
            str(item) for item in np.unique(list(self.y_train) + list(self.y_test))
        ]

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

        return metrics.accuracy_score(self.y_test, self.y_pred, **kwargs)

    def balanced_accuracy(self, **kwargs):
        """
        The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets.
        It is defined as the average of recall obtained on each class.

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

        return metrics.balanced_accuracy_score(self.y_test, self.y_pred, **kwargs)

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
            return metrics.average_precision_score(
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
            roc_auc = metrics.roc_auc_score(
                self.y_test, self.probabilities, multi_class=multi_class, **kwargs
            )
        else:
            if hasattr(self.model, "decision_function"):
                roc_auc = metrics.roc_auc_score(
                    self.y_test, self.model.decision_function(self.x_test), **kwargs
                )
            else:
                roc_auc = np.nan

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

        return metrics.zero_one_loss(self.y_test, self.y_pred, **kwargs)

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
            return metrics.recall_score(self.y_test, self.y_pred, average=avg, **kwargs)
        else:
            return metrics.recall_score(self.y_test, self.y_pred, **kwargs)

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
            return metrics.precision_score(
                self.y_test, self.y_pred, average=avg, **kwargs
            )
        else:
            return metrics.precision_score(self.y_test, self.y_pred, **kwargs)

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

        return metrics.matthews_corrcoef(self.y_test, self.y_pred, **kwargs)

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
            return metrics.log_loss(self.y_test, self.probabilities, **kwargs)
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
            return metrics.jaccard_score(
                self.y_test, self.y_pred, average=avg, **kwargs
            )
        else:
            return metrics.jaccard_score(self.y_test, self.y_pred, **kwargs)

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
            return metrics.hinge_loss(
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

        return metrics.hamming_loss(self.y_test, self.y_pred, **kwargs)

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
            return metrics.fbeta_score(
                self.y_test, self.y_pred, beta, average=avg, **kwargs
            )
        else:
            return metrics.fbeta_score(self.y_test, self.y_pred, beta, **kwargs)

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
            return metrics.f1_score(self.y_test, self.y_pred, average=avg, **kwargs)
        else:
            return metrics.f1_score(self.y_test, self.y_pred, **kwargs)

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

        return metrics.cohen_kappa_score(self.y_test, self.y_pred, **kwargs)

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

        return metrics.brier_score_loss(self.y_test, self.y_pred, **kwargs)

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

        from aethos.model_analysis.constants import CLASS_METRICS_DESC

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

        import seaborn as sns

        y_true = self.y_test
        y_pred = self.y_pred

        if figsize:
            plt.figure(figsize=figsize)

        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

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

        if output_file or _global_config["track_experiments"]:  # pragma: no cover
            heatmap.figure.savefig(
                os.path.join(IMAGE_DIR, self.model_name, output_file)
            )

        if _global_config["track_experiments"]:  # pragma: no cover
            track_artifacts(self.run_id, self.model_name)

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

            roc_plot = metrics.plot_roc_curve(self.model, self.x_test, self.y_test)
            roc_plot.ax_.set_xlabel("False Positive Rate or (1 - Specifity)")
            roc_plot.ax_.set_ylabel("True Positive Rate or (Sensitivity)")
            if title:
                roc_plot.figure_.suptitle("ROC Curve (area = {:.2f})".format(roc_auc))

        if output_file:  # pragma: no cover
            roc_plot.figure_.savefig(
                os.path.join(IMAGE_DIR, self.model_name, output_file)
            )

        if _global_config["track_experiments"]:  # pragma: no cover
            track_artifacts(self.run_id, self.model_name)

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

        classification_report = metrics.classification_report(
            self.y_test, self.y_pred, target_names=self.classes, digits=2
        )

        print(classification_report)

    def cross_validate(
        self,
        cv_type="strat-kfold",
        score="accuracy",
        n_splits=5,
        shuffle=False,
        **kwargs
    ):
        """
        Runs cross validation on a Classification model.

        Scoring Metrics:
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’'
            - ‘roc_auc_ovr’
            - ‘roc_auc_ovo’
            - ‘roc_auc_ovr_weighted’
            - ‘roc_auc_ovo_weighted’
        
        Parameters
        ----------
        cv_type : {kfold, strat-kfold}, optional
            Crossvalidation type, by default "kfold"

        score : str, optional
            Scoring metric, by default "accuracy"

        n_splits : int, optional
            Number of times to split the data, by default 5

        shuffle : bool, optional
            True to shuffle the data, by default False
        """

        super()._cross_validate(cv_type, score, n_splits, shuffle, **kwargs)

    def decision_boundary(self, x=None, y=None, title="Decisioun Boundary"):
        """
        Plots a decision boundary for a given model.

        If no x or y columns are provided, it defaults to the first 2 columns of your data.

        Parameters
        ----------
        x : str, optional
            Column in the dataframe to plot, Feature one, by default None

        y : str, optional
            Column in the dataframe to plot, Feature two, by default None

        title : str, optional
            Title of the decision boundary plot, by default "Decisioun Boundary"
        """

        from yellowbrick.contrib.classifier import DecisionViz
        from sklearn.base import clone

        assert (not x or not y) or (
            x or y
        ), "Both x and y (feature 1 and 2) must be provided"

        if not x:
            features = [self.train_results.columns[0], self.train_results.columns[1]]
        else:
            features = [x, y]

        model = clone(self.model)
        viz = DecisionViz(model, title=title, features=features, classes=self.classes)

        viz.fit(self.x_train[features].values, self.y_train.values)
        viz.draw(self.x_test[features].values, self.y_test.values)
        viz.show()
