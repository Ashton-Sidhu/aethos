import warnings
import pandas as pd
import sklearn
import copy


from IPython.display import display
from ipywidgets import widgets
from ipywidgets.widgets.widget_layout import Layout

from aethos.config import shell
from aethos.config.config import _global_config
from aethos.model_analysis.unsupervised_model_analysis import UnsupervisedModelAnalysis
from aethos.modelling.util import (
    _get_cv_type,
    _make_img_project_dir,
    _run_models_parallel,
    run_gridsearch,
    to_pickle,
    track_model,
)
from aethos.util import split_data, _get_attr_, _get_item_

warnings.simplefilter("ignore", FutureWarning)


class ModelBase(object):
    def __init__(
        self,
        x_train,
        target,
        x_test=None,
        test_split_percentage=0.2,
        exp_name="my-experiment",
    ):

        self._models = {}
        self._queued_models = {}
        self.exp_name = exp_name

        problem = "c" if type(self).__name__ == "Classification" else "r"

        self.x_train = x_train
        self.x_test = x_test
        self.target = target
        self.test_split_percentage = test_split_percentage
        self.target_mapping = None

        if self.x_test is None and type(self).__name__ != "Unsupervised":
            # Generate train set and test set.
            self.x_train, self.x_test = split_data(
                self.x_train, test_split_percentage, self.target, problem
            )
            self.x_train = self.x_train.reset_index(drop=True)
            self.x_test = self.x_test.reset_index(drop=True)

    def __getitem__(self, key):

        return _get_item_(self, key)

    def __getattr__(self, key):

        # For when doing multi processing when pickle is reconstructing the object
        if key in {"__getstate__", "__setstate__"}:
            return object.__getattr__(self, key)

        return self._models[key] if key in self._models else _get_attr_(self, key)

    def __setattr__(self, key, value):

        if key not in self.__dict__ or hasattr(self, key):
            # any normal attributes are handled normally
            dict.__setattr__(self, key, value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):

        if key in self.__dict__:
            dict.__setitem__(self.__dict__, key, value)

    def __repr__(self):

        return self.x_train.head().to_string()

    def _repr_html_(self):  # pragma: no cover

        cols = self.features + [self.target] if self.target else self.features
        return self.x_train[cols].head()._repr_html_()

    def __deepcopy__(self, memo):

        x_test = self.x_test.copy() if self.x_test is not None else None

        new_inst = type(self)(
            x_train=self.x_train.copy(),
            target=self.target,
            x_test=x_test,
            test_split_percentage=self.test_split_percentage,
            exp_name=self.exp_name,
        )

        new_inst.target_mapping = self.target_mapping
        new_inst._models = self._models
        new_inst._queued_models = self._queued_models

        return new_inst

    @property
    def features(self):
        """Features for modelling"""

        cols = self.x_train.columns.tolist()

        if self.target:
            cols.remove(self.target)

        return cols

    @property
    def train_data(self):
        """Training data used for modelling"""

        return self.x_train[self.features]

    @train_data.setter
    def train_data(self, val):
        """Setting for train_data"""

        val[self.target] = self.y_train
        self.x_train = val

    @property
    def test_data(self):
        """Testing data used to evaluate models"""

        return self.x_test[self.features] if self.x_test is not None else None

    @test_data.setter
    def test_data(self, val):
        """Test data setter"""

        val[self.target] = self.y_test
        self.x_test = val

    @property
    def y_test(self):
        """
        Property function for the testing predictor variable
        """

        if self.x_test is not None and self.target:
            return self.x_test[self.target]
        else:
            return None

    @y_test.setter
    def y_test(self, value):
        """
        Setter function for the testing predictor variable
        """

        if self.x_test is not None:
            if self.target:
                self.x_test[self.target] = value
            else:
                self.target = "label"
                self.x_test["label"] = value
                print('Added a target (predictor) field (column) named "label".')

    @property
    def columns(self):
        """
        Property to return columns in the dataset.
        """

        return self.x_train.columns.tolist()

    def copy(self):
        """
        Returns deep copy of object.
        
        Returns
        -------
        Object
            Deep copy of object
        """

        return copy.deepcopy(self)

    def help_debug(self):
        """
        Displays a tips for helping debugging model outputs and how to deal with over and underfitting.

        Credit: Andrew Ng's and his book Machine Learning Yearning

        Examples
        --------
        >>> model.help_debug()
        """

        from aethos.model_analysis.constants import DEBUG_OVERFIT, DEBUG_UNDERFIT

        overfit_labels = [
            widgets.Checkbox(description=item, layout=Layout(width="100%"))
            for item in DEBUG_OVERFIT
        ]
        underfit_labels = [
            widgets.Checkbox(description=item, layout=Layout(width="100%"))
            for item in DEBUG_UNDERFIT
        ]

        overfit_box = widgets.VBox(overfit_labels)
        underfit_box = widgets.VBox(underfit_labels)

        tab_list = [overfit_box, underfit_box]

        tab = widgets.Tab()
        tab.children = tab_list
        tab.set_title(0, "Overfit")
        tab.set_title(1, "Underfit")

        display(tab)

    def run_models(self, method="parallel"):
        """
        Runs all queued models.

        The models can either be run one after the other ('series') or at the same time in parallel.

        Parameters
        ----------
        method : str, optional
            How to run models, can either be in 'series' or in 'parallel', by default 'parallel'

        Examples
        --------
        >>> model.run_models()
        >>> model.run_models(method='series')
        """

        models = []

        if method == "parallel":
            models = _run_models_parallel(self)
        elif method == "series":
            for model in self._queued_models:
                models.append(self._queued_models[model]())
        else:
            raise ValueError(
                'Invalid run method, accepted run methods are either "parallel" or "series".'
            )

        return models

    def list_models(self):
        """
        Prints out all queued and ran models.

        Examples
        --------
        >>> model.list_models()
        """

        print("######## QUEUED MODELS ########")
        if self._queued_models:
            for key in self._queued_models:
                print(key)
        else:
            print("No queued models.")

        print()

        print("######### RAN MODELS ##########")
        if self._models:
            for key in self._models:
                print(key)
        else:
            print("No ran models.")

    def delete_model(self, name):
        """
        Deletes a model, specified by it's name - can be viewed by calling list_models.

        Will look in both queued and ran models and delete where it's found.

        Parameters
        ----------
        name : str
            Name of the model

        Examples
        --------
        >>> model.delete_model('model1')
        """

        if name in self._queued_models:
            del self._queued_models[name]
        elif name in self._models:
            del self._models[name]
        else:
            raise ValueError(f"Model {name} does not exist")

        self.list_models()

    def compare_models(self):
        """
        Compare different models across every known metric for that model.
        
        Returns
        -------
        Dataframe
            Dataframe of every model and metrics associated for that model
        
        Examples
        --------
        >>> model.compare_models()
        """

        results = [self._models[model].metrics() for model in self._models]

        results_table = pd.concat(results, axis=1, join="inner")
        results_table = results_table.loc[:, ~results_table.columns.duplicated()]

        # Move descriptions column to end of dataframe.
        descriptions = results_table.pop("Description")
        results_table["Description"] = descriptions

        return results_table

    def to_pickle(self, name: str):
        """
        Writes model to a pickle file.
        
        Parameters
        ----------
        name : str
            Name of the model

        Examples
        --------
        >>> m = Model(df)
        >>> m.LogisticRegression()
        >>> m.to_pickle('log_reg')
        """

        model_obj = self._models[name]

        to_pickle(model_obj.model, model_obj.model_name)

    def _run_supervised_model(
        self,
        model,
        model_name,
        model_type,
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        run=True,
        verbose=1,
        **kwargs,
    ):
        """
        Helper function that generalizes model orchestration.
        """

        #############################################################
        ################## Initialize Variables #####################
        #############################################################

        # Hard coding SVR due to its parent having random_state and the child not allowing it.
        random_state = kwargs.pop("random_state", None)
        if (
            not random_state
            and hasattr(model(), "random_state")
        ):
            random_state = 42

        run_id = None

        _make_img_project_dir(model_name)

        #############################################################
        #################### Initialize Model #######################
        #############################################################

        if random_state:
            model = model(random_state=random_state, **kwargs)
        else:
            model = model(**kwargs)

        #############################################################
        ####################### Gridsearch ##########################
        #############################################################

        if gridsearch:
            grid_cv = _get_cv_type(cv_type, 5, False) if cv_type is not None else 5

            model = run_gridsearch(model, gridsearch, grid_cv, score, verbose=verbose)

        #############################################################
        ###################### Train Model ##########################
        #############################################################

        # Train a model and predict on the test test.
        model.fit(self.train_data, self.y_train)

        #############################################################
        ############### Initialize Model Analysis ###################
        #############################################################

        if gridsearch:
            model = model.best_estimator_

        self._models[model_name] = model_type(
            model, self.x_train, self.x_test, self.target, model_name,
        )

        #############################################################
        ######################## Tracking ###########################
        #############################################################

        if _global_config["track_experiments"]:  # pragma: no cover
            if random_state is not None:
                kwargs["random_state"] = random_state

            run_id = track_model(
                self.exp_name,
                model,
                model_name,
                kwargs,
                self.compare_models()[model_name].to_dict(),
            )
            self._models[model_name].run_id = run_id

        print(model)

        return self._models[model_name]

    def _run_unsupervised_model(
        self, model, model_name, run=True, **kwargs,
    ):
        """
        Helper function that generalizes model orchestration.
        """

        #############################################################
        ################## Initialize Variables #####################
        #############################################################

        # Hard coding OneClassSVM due to its parent having random_state and the child not allowing it.
        random_state = kwargs.pop("random_state", None)
        if (
            not random_state
            and "random_state" in dir(model())
            and not isinstance(model(), sklearn.svm.OneClassSVM)
        ):
            random_state = 42

        _make_img_project_dir(model_name)

        #############################################################
        #################### Initialize Model #######################
        #############################################################

        if random_state:
            model = model(random_state=random_state, **kwargs)
        else:
            model = model(**kwargs)

        #############################################################
        ###################### Train Model ##########################
        #############################################################

        model.fit(self.train_data)

        #############################################################
        ############### Initialize Model Analysis ###################
        #############################################################

        self._models[model_name] = UnsupervisedModelAnalysis(
            model, self.x_train, model_name
        )

        #############################################################
        ######################## Tracking ###########################
        #############################################################

        if _global_config["track_experiments"]:  # pragma: no cover
            if random_state is not None:
                kwargs["random_state"] = random_state

            track_model(self.exp_name, model, model_name, kwargs)

        print(model)

        return self._models[model_name]
