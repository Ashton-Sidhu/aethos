import aethos.config.config as cf
from aethos.config import shell
from aethos.config.config import is_bool, is_list

interactive_df_doc = """
: bool
    Use QGrid library to interact with pandas dataframe.
    Default value is False
    Valid values: False, True
"""

interactive_table_doc = """
: bool
    Use iTable library to interact with pandas dataframe.
    Default value is False
    Valid values: False, True
"""

project_metric_doc = """
: list
    Set a default project metric to evaluate models against.
    Default value is []
    Valid values:
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

        'Explained Variance': 'Explained variance regression score function. Best possible score is 1.0, lower values are worse.',
        
        'Max Error': 'Returns the single most maximum residual error.',
        
        'Mean Absolute Error': 'Postive mean value of all residuals',
        
        'Mean Squared Error': 'Mean of the squared sum the residuals',
        
        'Root Mean Sqaured Error': 'Square root of the Mean Squared Error',
        
        'Mean Squared Log Error': 'Mean of the squared sum of the log of all residuals',
        
        'Median Absolute Error': 'Postive median value of all residuals',
        
        'R2': 'R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable that is explained by an independent variable or variables in a regression model.',
        
        'SMAPE': 'Symmetric mean absolute percentage error. It is an accuracy measure based on percentage (or relative) errors.
    """

word_doc = """
: bool
    Write report in a word file
    Default value is False
    Valid values: False, True
"""


def use_qgrid(key):
    import qgrid

    if shell == "ZMQInteractiveShell":
        if cf.get_option(key):
            qgrid.enable()
            qgrid.set_defaults(show_toolbar=True)
        else:
            qgrid.disable()


def use_itable(key):
    import itables.interactive
    import itables.options as opt

    opt.lengthMenu = [5, 10, 20, 50, 100, 200, 500]


cf.register_option(
    "interactive_df",
    default=False,
    doc=interactive_df_doc,
    validator=is_bool,
    cb=use_qgrid,
)

cf.register_option(
    "interactive_table",
    default=False,
    doc=interactive_table_doc,
    validator=is_bool,
    cb=use_itable,
)

cf.register_option(
    "project_metrics", default=[], doc=project_metric_doc, validator=is_list
)

cf.register_option(
    "word_report", default=False, doc=word_doc, validator=is_bool,
)
