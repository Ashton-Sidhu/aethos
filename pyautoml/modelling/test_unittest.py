import unittest

import numpy as np
import pandas as pd

from pyautoml import Model


class TestModelling(unittest.TestCase):

    def test_text_gensim_summarize(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(x_train=data, split=False)
        model.summarize_gensim('data', ratio=0.5, run=True)
        validate = model.data_summarized is not None

        self.assertTrue(validate)

    
    def test_text_gensim_keywords(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(x_train=data, split=False)
        model.extract_keywords_gensim('data', ratio=0.5, run=True)
        validate = model.data_extracted_keywords is not None

        self.assertTrue(validate)

    def test_text_gensim_w2v(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(x_train=data, split=False)
        model.word2vec('data', prep=True, run=True, min_count=1)
        validate = model.w2v is not None

        self.assertTrue(validate)

    def test_text_w2vprep(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])
        data['prep'] = pd.Series([text.split() for text in text_data])

        model = Model(x_train=data, split=False)
        model.word2vec('prep', run=True, min_count=1)
        validate = model.w2v is not None

        self.assertTrue(validate)

    def test_text_d2v(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(x_train=data, split=False)
        model.doc2vec('data', prep=True, run=True, min_count=1)
        validate = model.d2v is not None

        self.assertTrue(validate)

    def test_text_d2vprep(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])
        data['prep'] = pd.Series([text.split() for text in text_data])

        model = Model(x_train=data, split=False)
        model.doc2vec('prep', run=True, min_count=1)
        validate = model.d2v is not None

        self.assertTrue(validate)

    def test_model_getattr(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(x_train=data, split=False)
        model.extract_keywords_gensim('data', ratio=0.5, model_name='model1', run=True)
        validate = model.model1 is not None and model['model1'] is not None

        self.assertTrue(validate)

    def test_model_addtoqueue(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(x_train=data, split=False)
        model.extract_keywords_gensim('data', ratio=0.5, model_name='model1', run=False)
        model.summarize_gensim('data', ratio=0.5, run=False)
        validate = len(model._queued_models)

        self.assertEqual(validate, 2)

    def test_model_kmeans(self):

        data = [[1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2'])

        model = Model(x_train=data, split=False)
        model.kmeans(n_clusters=3, random_state=0, run=True)
        validate = model.kmeans_clusters is not None

        self.assertTrue(validate)

    def test_model_kmeans_split(self):

        data = [[1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2'])

        model = Model(x_train=data)
        model.kmeans(n_clusters=3, random_state=0, run=True)
        validate = model.x_train_results.kmeans_clusters is not None and model.x_test_results.kmeans_clusters is not None

        self.assertTrue(validate)

    def test_model_dbscan(self):

        data = [[1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2'])

        model = Model(x_train=data, split=False)
        model.dbscan(eps=3, min_samples=2, run=True)
        validate = model.dbscan_clusters is not None

        self.assertTrue(validate)

    def test_model_cluster_filter(self):

        data = [[1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2'])

        model = Model(x_train=data, split=False)
        model = model.dbscan(eps=3, min_samples=2, run=True)
        filtered = model.filter_cluster(0)
        validate = all(filtered.dbscan_clusters == 0)

        self.assertTrue(validate)

    def test_model_unsupervised_defaultgridsearch(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', report_name='gridsearch_test')
        
        gridsearch_params = {
            'k': [1, 2]
        }
        model.kmeans(gridsearch=gridsearch_params, cv=2, run=True)

        self.assertTrue(True)

    def test_model_defaultgridsearch(self):

        data = [[1, 2, 1], [2, 2, 1], [2, 3, 1],
            [8, 7, 0], [8, 8, 0], [25, 80, 0],
            [1, 2, 1], [3, 2, 0], [1, 2, 1]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', report_name='gridsearch_test')
        
        gridsearch_params = {
            'C': [0.2, 1]
        }
        model.logistic_regression(gridsearch=gridsearch_params, cv=2, run=True)

        self.assertTrue(True)

    def test_model_logisticregression(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        validate = model.x_train_results.log_predictions is not None and model.x_test_results.log_predictions is not None

        self.assertTrue(validate)

    def test_model_confusionmatrix(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.confusion_matrix()

        self.assertTrue(True)

    def test_model_report_confusionmatrix(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', report_name='confusion_report')
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.confusion_matrix()

        self.assertTrue(True)

    def test_model_all_score_metrics(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', report_name='metric_report')
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.metrics()

        self.assertTrue(True)

    def test_model_report_classificationreport(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', report_name='classification_report')
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.classification_report()

        self.assertTrue(True)

    def test_model_report_modelweights(self):
        
        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', report_name='modelweights')
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.model_weights()

        self.assertTrue(True)

    def test_plot_roccurve(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5, report_name='modelweights')
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.roc_curve()

        self.assertTrue(True)
    
    def test_decision_plot(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5)
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.decision_plot()

        self.assertTrue(True)

    def test_decision_plot_all(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5)
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.decision_plot(num_samples='all')

        self.assertTrue(True)

    def test_decision_plot_sameaxis(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5)
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        r = model.log_reg.decision_plot(sample_no=1)
        model.log_reg.decision_plot(sample_no=2, feature_order=r.feature_idx, xlim=r.xlim)

        self.assertTrue(True)

    def test_decision_plot_misclassified(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5)
        model.logistic_regression(random_state=2, run=True)
        model.log_reg.decision_plot(0.75, highlight_misclassified=True)

        self.assertTrue(True)

    def test_force_plot(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5)
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.force_plot()

        self.assertTrue(True)

    def test_force_plot_misclassified(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.6)
        model.logistic_regression(random_state=2, run=True)
        model.log_reg.force_plot(misclassified=True)

        self.assertTrue(True)

    def test_get_misclassified(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5)
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.shap_get_misclassified_index()

        self.assertTrue(True)

    def test_summaryplot(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5)
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.summary_plot()

        self.assertTrue(True)

    def test_dependence_plot(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5)
        model.logistic_regression(random_state=2, penalty='l2', run=True)
        model.log_reg.dependence_plot('col1')

        self.assertTrue(True)    

    def test_local_multiprocessing(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5, report_name='modelweights')
        model.logistic_regression(random_state=2, penalty='l2', model_name='l1', run=True)
        model.logistic_regression(random_state=2, penalty='l2', model_name='l2', run=True)
        model.logistic_regression(random_state=2, penalty='l2', model_name='l3', run=True)

        model.run_models()

        self.assertTrue(len(model._models) == 3 and len(model._queued_models) == 0)

    def test_local_seriesprocessing(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5, report_name='modelweights')
        model.logistic_regression(random_state=2, penalty='l2', model_name='l1', run=True)
        model.logistic_regression(random_state=2, penalty='l2', model_name='l2', run=True)
        model.logistic_regression(random_state=2, penalty='l2', model_name='l3', run=True)

        model.run_models(method='series')

        self.assertTrue(len(model._models) == 3 and len(model._queued_models) == 0)

    def test_interpretmodel_behaviour_all(self):


        train_data = np.random.random_sample(size=(1000,2))
        label_data = np.random.randint(0, 2, size=(1000,1))

        data = pd.DataFrame(data=train_data, columns=['col1', 'col2'])
        data['col3'] = label_data

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.2)
        model.logistic_regression(random_state=2, run=True)
        model.log_reg.interpret_model_behavior(show=False)

        self.assertTrue(True)

    def test_interpretmodel_behaviour_dependence(self):

        data = np.random.randint(0, 2, size=(1000,3))
        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.4)
        model.logistic_regression(random_state=2, run=True)
        model.log_reg.interpret_model_behavior(method='dependence', show=False)

        self.assertTrue(True)

    def test_interpretmodel_predictions_all(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.6)
        model.logistic_regression(random_state=2, run=True)
        model.log_reg.interpret_predictions(show=False)

        self.assertTrue(True)

    def test_interpretmodel_predictions_lime(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.6)
        model.logistic_regression(random_state=2, run=True)
        model.log_reg.interpret_predictions(method='lime', show=False)

        self.assertTrue(True)

    def test_interpretmodel_performance_all(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.6)
        model.logistic_regression(random_state=2, run=True)
        model.log_reg.interpret_model_performance(show=False)

        self.assertTrue(True)

    def test_interpretmodel_performance_roc(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.6)
        model.logistic_regression(random_state=2, run=True)
        model.log_reg.interpret_model_performance(method='ROC', show=False)

        self.assertTrue(True)

    def test_interpret_model(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.4)
        model.logistic_regression(random_state=2, run=True)
        model.log_reg.interpret_model(show=False)

        self.assertTrue(True)

    def test_interpret_model_prerun(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.4)
        model.logistic_regression(random_state=2, run=True)
        model.log_reg.interpret_model_performance(method='ROC', show=False)
        model.log_reg.interpret_model(show=False)

        self.assertTrue(True)

    def test_comparemodels(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3', test_split_percentage=0.5, report_name='modelweights')
        model.logistic_regression(random_state=2, penalty='l2', model_name='l1', run=True)
        model.logistic_regression(random_state=2, penalty='l2', model_name='l2', run=True)
        model.logistic_regression(random_state=2, penalty='l2', model_name='l3', run=True)

        model.run_models(method='series')
        model.compare_models()

        self.assertTrue(True)

    def test_cv(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])
        model = Model(x_train=data, target_field='col3', test_split_percentage=0.2)
        model.logistic_regression(cv=2, random_state=2, learning_curve=True)

        self.assertTrue(True)

    def test_unsupervisedcv(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])
        model = Model(x_train=data, target_field='col3', test_split_percentage=0.2)
        model.kmeans(cv=2, random_state=2, learning_curve=True)

        self.assertTrue(True)

    def test_stratified_cv(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])
        model = Model(x_train=data, target_field='col3', test_split_percentage=0.2)
        cv_values = model.logistic_regression(cv='strat-kfold', random_state=2, learning_curve=True, run=False)

        self.assertIsNotNone(len(cv_values) == 5)

    def test_del_model(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])
        model = Model(x_train=data, target_field='col3', test_split_percentage=0.2)
        model.logistic_regression(random_state=2, run=True)
        model.delete_model('log_reg')

        self.assertTrue(len(model._models) == 0)

    def test_model_ridgeclassifier(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.ridge_classification(random_state=2, run=True)
        validate = model.ridge_cls is not None

        self.assertTrue(validate)

    def test_model_sgdclassifier(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.sgd_classification(random_state=2, run=True)
        validate = model.sgd_cls is not None

        self.assertTrue(validate)

    def test_model_adaclassifier(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.adaboost_classification(random_state=2, run=True)
        validate = model.ada_cls is not None

        self.assertTrue(validate)

    def test_model_bagclassifier(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.bagging_classification(random_state=2, run=True)
        validate = model.bag_cls is not None

        self.assertTrue(validate)

    def test_model_boostingclassifier(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.gradient_boosting_classification(random_state=2, run=True)
        validate = model.grad_cls is not None

        self.assertTrue(validate)

    def test_model_isoforest(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data)
        model.isolation_forest(random_state=2, run=True)
        validate = model.iso_forest is not None

        self.assertTrue(validate)

    def test_model_oneclasssvm(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data)
        model.oneclass_svm(run=True)
        validate = model.ocsvm is not None

        self.assertTrue(validate)

    def test_model_rfclassifier(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.random_forest_classification(random_state=2, run=True)
        validate = model.rf_cls is not None

        self.assertTrue(validate)

    def test_model_bernoulli(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.nb_bernoulli_classification(run=True)
        validate = model.bern is not None

        self.assertTrue(validate)

    def test_model_gaussian(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.nb_gaussian_classification(run=True)
        validate = model.gauss is not None

        self.assertTrue(validate)

    def test_model_multinomial(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.nb_multinomial_classification(run=True)

        validate = model.multi is not None

        self.assertTrue(validate)

    def test_model_dtclassifier(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.decision_tree_classification(random_state=2, run=True)
        validate = model.dt_cls is not None

        self.assertTrue(validate)

    def test_model_linearsvc(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.linearsvc(random_state=2, run=True)
        validate = model.linsvc is not None

        self.assertTrue(validate)

    def test_model_svc(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.svc(random_state=2, run=True)
        validate = model.svc_cls is not None

        self.assertTrue(validate)

    def test_model_bayesianridge(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.bayesian_ridge_regression(random_state=2, run=True)
        validate = model.bayridge_reg is not None

        self.assertTrue(validate)
    
    def test_model_elasticnet(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.elasticnet_regression(random_state=2, run=True)
        validate = model.elastic is not None

        self.assertTrue(validate)
    
    def test_model_lasso(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.lasso_regression(random_state=2, run=True)
        validate = model.lasso is not None

        self.assertTrue(validate)

    def test_model_linreg(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.linear_regression(random_state=2, run=True)
        validate = model.lin_reg is not None

        self.assertTrue(validate)

    def test_model_ridgeregression(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.ridge_regression(random_state=2, run=True)
        validate = model.ridge_reg is not None

        self.assertTrue(validate)

    def test_model_sgdregression(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.sgd_regression(random_state=2, run=True)
        validate = model.sgd_reg is not None

        self.assertTrue(validate)
        
    def test_model_adaregression(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.adaboost_regression(random_state=2, run=True)
        validate = model.ada_reg is not None

        self.assertTrue(validate)
    
    def test_model_bgregression(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.bagging_regression(random_state=2, run=True)
        validate = model.bag_reg is not None

        self.assertTrue(validate)

    def test_model_gbregression(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.gradient_boosting_regression(random_state=2, run=True)
        validate = model.grad_reg is not None

        self.assertTrue(validate)

    def test_model_rfregression(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.random_forest_regression(random_state=2, run=True)
        validate = model.rf_reg is not None

        self.assertTrue(validate)

    def test_model_dtregression(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.decision_tree_regression(random_state=2, run=True)
        validate = model.dt_reg is not None

        self.assertTrue(validate)

    def test_model_linearsvr(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.linearsvr(random_state=2, run=True)
        validate = model.linearsvr is not None

        self.assertTrue(validate)

    def test_model_svr(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.svr(run=True)
        validate = model.svr_reg is not None

        self.assertTrue(validate)

    def test_model_xgbc(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.xgboost_classification(run=True)
        validate = model.xgb_cls is not None

        self.assertTrue(validate)

    def test_model_xgbr(self):

        data = np.random.randint(0, 2, size=(1000,3))

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        model = Model(x_train=data, target_field='col3')
        model.xgboost_regression(run=True)
        validate = model.xgb_reg is not None

        self.assertTrue(validate)

    def test_model_agglom(self):

        data = [[1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2'])

        model = Model(x_train=data, split=False)
        model.agglomerative_clustering(n_clusters=2, run=True)
        validate = model.agglom is not None

        self.assertTrue(validate)

    def test_model_meanshift(self):

        data = [[1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2'])

        model = Model(x_train=data, split=False)
        model.mean_shift(run=True)
        validate = model.mshift is not None

        self.assertTrue(validate)

    def test_model_gaussianmixture(self):

        data = [[1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2'])

        model = Model(x_train=data, split=False)
        model.gaussian_mixture_clustering(run=True)
        validate = model.gm_cluster is not None

        self.assertTrue(validate)

if __name__ == "__main__":
    unittest.main()
