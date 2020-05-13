
*"A collection of tools for Data Scientists and ML Engineers for them to focus less on how to do the analysis and instead worry about what are the best analytic tools that will help gain the most insights from their data."*

Welcome to aethos's documentation!
====================================
Aethos is a library/platform that automates your data science and analytical tasks at any stage in the pipeline. Aethos is, at its core, a uniform API that helps automate analytical techniques from various libaries such as pandas, sci-kit learn, gensim, etc. 

Aethos provides:

  - Automated data science cleaning, preprocessing, feature engineering and modelling techniques through one line of code
  - Automated visualizations through one line of code
  - Reusable code - no more copying code from notebook to notebook
  - Automated dependency and corpus management
  - Datascience project templates
  - Integrated 3rd party jupyter plugins to make analyzing data more friendly
  - Model analysis use cases - Confusion Matrix, ROC Curve, all metrics, decision tree plots, etc.
  - Model interpretability - Local through SHAP and LIME, global through Morris Sensitivity
  - Interactive checklists and tips to either remind or help you through your analysis.
  - Comparing train and test data distribution
  - Exporting trained models as a service (Generates the necessary code, files and folder structure)
  - Experiment tracking with MLFlow
  - Statistical tests - Anova, T-test, etc.
  - Pre-trained models - BERT, GPT2, etc.

Plus more coming soon such as:

  - Testing for model drift
  - Recommendation models
  - Parralelization through Dask and/or Spark
  - Uniform API for deep learning models and automated code and file generation jupyter notebook development, python file of your data pipeline.
  - Automated code and file generation for jupyter notebook development and a python file of your data pipeline.

Aethos makes it easy to PoC, experiment and compare different techniques and models from various libraries. From imputations, visualizations, scaling, dimensionality reduction, feature engineering to modelling, model results and model deployment - all done with a single, human readable, line of code!

Aethos utilizes other open source libraries to help enhance your analysis from enhanced stastical information, interactive visual plots or statistical tests and models - all your tools in one place, all accessible with one line of code.

For more info such as features, development plan, status and vision checkout the `Aethos github page <https://github.com/Ashton-Sidhu/aethos/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Usage
=====

Examples can be viewed `here <https://github.com/Ashton-Sidhu/aethos/tree/develop/examples/>`_.

To start, we need to import the ethos dependencies as well as pandas.

Before that, we can create a full data science folder structure by running :code:`aethos create` from the command line and follow the command prompts.

For a full list of methods please see the full docs or `TECHNIQUES.md <https://github.com/Ashton-Sidhu/aethos/blob/develop/TECHNIQUES.md/>`_.

Options
-------

To enable extensions, such as QGrid interactive filtering:

.. code:: python

    import aethos as at

    at.options.interactive_df = True

Currently the following options are:

  - `interactive_df`: Interactive grid with QGrid
  - `interactive_table`: Interactive grid with Itable - comes with built in client side searching
  - `project_metrics`: Setting project metrics
    - Project metrics is a metric or set of metrics to evaluate models.
  - `track_experiments`: Uses MLFlow to track models and experiments.

User options such as changing the directory where images and projects are saved can be edited in the config file. This is located at `USER_HOME`/.aethos/ .

This location is also the default location of where any images and projects are stored.

***New in 2.0***

The Data and Model objects no longer exist but instead there a multiple objects you can use with more of a purpose.

Analysis - Used to analyze, visualize and run statistical models (t-tests, anovas, etc.)

Classification - Used to analyze, visualize, run statistical models and train classification models.

Regression - Used to analyze, visualize, run statistical models and train regression models.

Unsupervised - Used to analyze, visualize, run statistical models and train unsupervised models.

ClassificationModelAnalysis - Used to analyze, interpret and visualize results of a Classification model.

RegressionModelAnalysis - Used to analyze, interpret and visualize results of a Regression model.

UnsupervisedModelAnalysis - Used to analyze, interpret and visualize results of a Unsupervised model.

TextModelAnalysis - Used to analyze, interpret and visualize results of a Text model.


Analysis
--------

.. code:: python

    import aethos as at
    import pandas as pd

    x_train = pd.read_csv('data/train.csv') # load data into pandas

    # Initialize Data object with training data
    # By default, if no test data (x_test) is provided, then the data is split with 20% going to the test set
    # 
    # Specify predictor field as 'Survived'
    df = at.Classification(x_train, target='Survived')

    df.x_train # View your training data
    df.x_test # View your testing data

    df # Glance at your training data

    df[df.Age > 25] # Filter the data

    df.x_train['new_col'] = [1,2] # This is the exact same as the either of code above
    df.x_test['new_col'] = [1,2]

    df.data_report(title='Titanic Summary', output_file='titanic_summary.html') # Automate EDA with pandas profiling with an autogenerated report

    df.describe() # Display a high level view of your data using an extended version of pandas describe

    df.describe_column('Fare') # Get indepth statistics about the 'Fare' column

    df.mean() # Run pandas functions on the aethos objects

    df.missing_data # View your missing data at anytime

    df.correlation_matrix() # Generate a correlation matrix for your training data

    df.predictive_power() # Calculates the predictive power of each variable

    df.autoviz() # Runs autoviz on the data and runs EDA on your data

    df.pairplot() # Generate pairplots for your training data features at any time

    df.checklist() # Will provide an iteractive checklist to keep track of your cleaning tasks

**NOTE:** One of the benefits of using ``aethos`` is that any method you apply on your train set, gets applied to your test dataset. For any method that requires fitting (replacing missing data with mean), the method is fit on the training data and then applied to the testing data to avoid data leakage.

.. code:: python

    # Replace missing values in the 'Fare' and 'Embarked' column with the most common values in each of the respective columns.
    df.replace_missing_mostcommon('Fare', 'Embarked')

    # To create a "checkpoint" of your data (i.e. if you just want to test this analytical method), assign it to a variable
    df.replace_missing_mostcommon('Fare', 'Embarked')

    # Replace missing values in the 'Age' column with a random value that follows the probability distribution of the 'Age' column in the training set. 
    df.replace_missing_random_discrete('Age')

    df.drop('Cabin') # Drop the cabin column

As you've started to notice, alot of tasks to df the data and to explore the data have been reduced down to one command, and are also customizable by providing the respective keyword arguments (see documentation).

.. code:: python

    # Create a barplot of the mean surivial rate grouped by age.
    df.barplot(x='Age', y='Survived', method='mean')

    # Plots a scatter plot of Age vs. Fare and colours the dots based off the Survived column.
    df.scatterplot(x='Age', y='Fare', color='Survived')

    # One hot encode the `Person` and `Embarked` columns and then drop the original columns
    df.onehot_encode('Person', 'Embarked', drop_col=True) 

Modelling
=========

Running a Single Model
----------------------

Models can be trained one at a time or multiple at a time. They can also be trained by passing in the params for the sklearn, xgboost, etc constructor, by passing in a gridsearch dictionary & params, cross validating with gridsearch & params.

After a model has been ran, it comes with use cases such as plotting RoC curves, calculating performance metrics, confusion matrices, SHAP plots, decision tree plots and other local and global model interpretability use cases.

.. code:: python

    lr_model = df.LogisticRegression(random_state=42) # Train a logistic regression model

    # Train a logistic regression model with gridsearch
    lr_model = df.LogisticRegression(gridsearch={'penalty': ['l1', 'l2']}, random_state=42)

    # Crossvalidate a a logistic regression model, displays the scores and the learning curve and builds the model
    lr_model = df.LogisticRegression()
    lr_model.cross_validate(cv_type="strat-kfold", n_splits=10) # default is strat-kfold for classification  problems

    # Build a Logistic Regression model with Gridsearch and then cross validates the best model using stratified K-Fold cross validation.
    lr_model = model.LogisticRegression(gridsearch={'penalty': ['l1', 'l2']}, cv_type="strat-kfold") 

    lr_model.help_debug() # Interface with items to check for to help debug your model.

    lr_model.metrics() # Views all metrics for the model
    lr_model.confusion_matrix()
    lr_model.decision_boundary()
    lr_model.roc_curve()

Running Multiple Models
-----------------------

.. code:: python

    # Add a Logistic Regression, Random Forest Classification and a XGBoost Classification model to the queue.
    lr = df.LogisticRegression(random_state=42, model_name='log_reg', run=False)
    rf = df.RandomForestClassification(run=False)
    xgbc = df.XGBoostClassification(run=False)

    df.run_models() # This will run all queued models in parallel
    df.run_models(method='series') # Run each model one after the other

    df.compare_models() # This will display each model evaluated against every metric

    # Every model is accessed by a unique name that is assiged when you run the model.
    # Default model names can be seen in the function header of each model.

    df.log_reg.confusion_matrix() # Displays a confusion matrix for the logistic regression model
    df.rf_cls.confusion_matrix() # Displays a confusion matrix for the random forest model

Using Pretrained Models
------------------------

Currently you can use pretrained models such as BERT, XLNet, AlBERT, etc. to calculate sentiment and answer questions.

.. code:: python

    df.pretrained_sentiment_analysis(`text_column`)

    # To answer questions, context for the question has to be supplied
    df.pretrained_question_answer(`context_column`, `question_column`)

Model Interpretability
----------------------

As mentioned in the Model section, whenever a model is trained you have access to use cases for model interpretability as well. There are prebuild SHAP usecases and an interactive dashboard that is equipped with LIME and SHAP for local model interpretability and Morris Sensitivity for global model interpretability.

.. code:: python

    lr_model = model.LogisticRegression(random_state=42)

    lr_model.summary_plot() # SHAP summary plot
    lr_model.force_plot() # SHAP force plot
    lr_model.decision_plot() # SHAP decision plot
    lr_model.dependence_plot() # SHAP depencence plot

    lr_model.interpret_model() # Creates an interactive dashboard to view LIME, SHAP, Morris Sensitivity and more for your model

Code Generation
---------------

Currently you are only able to export your model to be ran a service, and will be able to automatically generate the required files. The automatic creation of a data pipeline is still in progress.

.. code:: python

    lr_model.to_service('titanic')

Now navigate to 'your_home_folder'('~' on linux and Users/'your_user_name' on windows)/.aethos/projects/titanic/ and you will see the files needed to run the model as a service using FastAPI and uvicorn. 

Installation
============

:code:`pip install aethos`

To install the dependencies to use pretrained models such as BERT, XLNet, AlBERT, etc:

:code:`pip install aethos[ptmodels]`

To install associating corpora for nltk analysis:

:code:`aethos install-corpora`

To install and use the extensions such as `qgrid` for interactive filtering and analysis with DataFrames:

:code:`aethos enable-extensions`

Currently working on condas implementation.

To create a Data Science project run:

:code:`aethos create`

This will create a full folder strucuture for you to manage data, unit tests, experiments and source code.

If experiment tracking is enabled or if you want to start the MLFlow UI:

:code:`aethos mlflow-ui`

This will start the MLFlow UI in the directory where your Aethos experiemnts are run.
NOTE: This only works for local use of MLFLOW, if you are running MLFlow on a remote server, just start it on the remote server and enter the address in the `%HOME%/.aethos/config.yml` file.

Configuration
=============

By default the configuration file is located at :code:`%HOME%/.aethos/config.yml`.

You can use the configuration file to specify the full path of where to store reports, images, deployed projects and experiments.

Project Metrics
===============

Often in data science projects, you define a metric or metrics to evaluate how well your model performs.

By default when training a model and viewing the results, Aethos calculates all possible metrics for the problem type (Unsupervised, Text, Classification, Regression, etc.).

To change this behaviour it is recommended to set project metrics:

.. code:: python

    import aethos as at

    at.options.project_metrics = ['F1', 'Precision', 'Recall']

Now when comparing models or viewing metrics for models, only the F1 score, precision and recall metrics will be shown and consequently tracked if tracking is enabled.

The supported project metrics are the following:

Classification
--------------

    - Accuracy
    - Balanced Accuracy
    - Average Precision
    - ROC AUC
    - Zero One Loss
    - Precision
    - Recall
    - Matthews Correlation Coefficient
    - Log Loss
    - Jaccard
    - Hinge Loss
    - Hamming Loss
    - F-Beta
    - F1
    - Cohen Kappa
    - Brier Loss
    - Explained Variance

Regression
----------

    - Max Error
    - Mean Absolute Error
    - Mean Squared Error
    - Root Mean Sqaured Error
    - Mean Squared Log Error
    - Median Absolute Error
    - R2
    - SMAPE

Using MLFlow
=============

To start tracking experiments with MLFlow, enable it runing the following:

.. code:: python

    import aethos as at
    
    at.options.track_experiments = True

Now any models you train will be tracked with MLFlow against all metrics unless you set project metrics, MLFlow will then only track the project metrics.

.. code:: python

    at.options.project_metrics = ['F1', 'Precision', 'Recall']

To start the MLFlow UI in the directory your experiments are stored run:

:code:`aethos mlflow-ui`

Note: This only works for local use of MLFlow, if you are running MLFlow on a remote server, start it on the server and enter in the address in the Aethos config file at `%HOME%/.aethos/config.yml`.

Analysis API
========

.. automodule:: aethos.analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aethos.visualizations.visualizations
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aethos.cleaning
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aethos.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aethos.feature_engineering
   :members:
   :undoc-members:
   :show-inheritance:
   
.. automodule:: aethos.stats.stats
   :members:
   :undoc-members:
   :show-inheritance:

Model API
==========

.. automodule:: aethos.modelling.model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aethos.modelling.classification_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aethos.modelling.regression_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aethos.modelling.unsupervised_models
   :members:
   :undoc-members:
   :show-inheritance:

Model Analysis API
===================

.. automodule:: aethos.model_analysis.model_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aethos.model_analysis.classification_model_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aethos.model_analysis.regression_model_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aethos.model_analysis.unsupervised_model_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aethos.model_analysis.text_model_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Examples
========

Examples can be viewed `here <https://github.com/Ashton-Sidhu/aethos/tree/develop/examples/>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
