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

        if self.data is not None:
            self.target_data = model_object.target_data
            self.prediction_data = self.data[predictions_col]
        else:
            self.test_target_data = model_object.test_target_data
            self.prediction_data = self.test_data[predictions_col]

    # TODO: Confusion Matrix
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
