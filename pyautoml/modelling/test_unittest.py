import unittest

import pandas as pd
from pyautoml import Model


class TestModelling(unittest.TestCase):

    def test_text_gensim_summarize(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(data=data, split=False)
        model.summarize_gensim('data', ratio=0.5, run=True)
        validate = model.data_summarized is not None

        self.assertTrue(validate)

    
    def test_text_gensim_keywords(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(data=data, split=False)
        model.extract_keywords_gensim('data', ratio=0.5, run=True)
        validate = model.data_extracted_keywords is not None

        self.assertTrue(validate)


    def test_model_getattr(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(data=data, split=False)
        model.extract_keywords_gensim('data', ratio=0.5, model_name='model1', run=True)
        validate = model.model1 is not None and model['model1'] is not None

        self.assertTrue(validate)

    def test_model_addtoqueue(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(data=data, split=False)
        model.extract_keywords_gensim('data', ratio=0.5, model_name='model1', run=False)
        model.summarize_gensim('data', ratio=0.5, run=False)
        validate = len(model._queued_models)

        self.assertEquals(validate, 2)

    def test_model_kmeans(self):

        data = [[1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2'])

        model = Model(data=data, split=False)
        model.kmeans(n_clusters=3, random_state=0)
        validate = model.kmeans_clusters is not None

        self.assertTrue(validate)

    
    def test_model_kmeans_split(self):

        data = [[1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2'])

        model = Model(data=data)
        model.kmeans(n_clusters=3, random_state=0)
        validate = model.train_data.kmeans_clusters is not None and model.test_data.kmeans_clusters is not None

        self.assertTrue(validate)

    def test_model_dbscan(self):

        data = [[1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2'])

        model = Model(data=data, split=False)
        model.dbscan(eps=3, min_samples=2)
        validate = model.dbscan_clusters is not None

        self.assertTrue(validate)

    def test_model_cluster_filter(self):

        data = [[1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2'])

        model = Model(data=data, split=False)
        model = model.dbscan(eps=3, min_samples=2)
        filtered = model.filter_cluster(0)
        validate = all(filtered.dbscan_clusters == 0)

        self.assertTrue(validate)

    def test_model_defaultgridsearch(self):

        data = [[1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0],
            [1, 2, 1], [3, 2, 0], [1, 2, 1]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(data=data, target_field='col3', report_name='gridsearch_test')
        model.logistic_regression(gridsearch=True, gridsearch_cv=2)

        self.assertTrue(True)

    def test_model_logisticregression(self):

        data = [[1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(data=data, target_field='col3')
        model.logistic_regression(random_state=2, penalty='l1')
        validate = model.train_data.log_predictions is not None and model.test_data.log_predictions is not None

        self.assertTrue(validate)

    def test_model_confusionmatrix(self):

        data = [[1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(data=data, target_field='col3')
        model.logistic_regression(random_state=2, penalty='l1')
        model.log_reg.confusion_matrix()

        self.assertTrue(True)

    def test_model_report_confusionmatrix(self):

        data = [[1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(data=data, target_field='col3', report_name='confusion_report')
        model.logistic_regression(random_state=2, penalty='l1')
        model.log_reg.confusion_matrix()

        self.assertTrue(True)

    def test_model_all_score_metrics(self):

        data = [[1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0],
            [1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0],
            [1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0],
            [1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(data=data, target_field='col3', report_name='metric_report')
        model.logistic_regression(random_state=2, penalty='l1')
        model.log_reg.metric('all', metric='all')

        self.assertTrue(True)

    def test_model_report_classificationreport(self):

        data = [[1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(data=data, target_field='col3', report_name='classification_report')
        model.logistic_regression(random_state=2, penalty='l1')
        model.log_reg.classification_report()

        self.assertTrue(True)

    def test_model_report_modelweights(self):

        data = [[1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(data=data, target_field='col3', report_name='modelweights')
        model.logistic_regression(random_state=2, penalty='l1')
        model.log_reg.model_weights()

        self.assertTrue(True)

    def test_plot_roccurve(self):

        data = [[1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0],
            [1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0],
            [1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0],
            [1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(data=data, target_field='col3', test_split_percentage=0.5, report_name='modelweights')
        model.logistic_regression(random_state=2, penalty='l1')
        model.log_reg.roc_curve()

        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
