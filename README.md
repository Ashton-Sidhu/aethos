[![PyPI version](https://badge.fury.io/py/aethos.svg)](https://badge.fury.io/py/aethos) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aethos) [![CircleCI](https://circleci.com/gh/Ashton-Sidhu/aethos/tree/develop.svg?style=svg)](https://circleci.com/gh/Ashton-Sidhu/aethos/tree/develop) [![Documentation Status](https://readthedocs.org/projects/aethos/badge/?version=latest)](https://aethos.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/Ashton-Sidhu/aethos/branch/develop/graph/badge.svg)](https://codecov.io/gh/Ashton-Sidhu/aethos)



# Aethos

<i>"A collection of tools for Data Scientists and ML Engineers to automate their workflow of performing analysis to deploying models and pipelines."</i>

To track development of the project, you can view the [Trello board](https://trello.com/b/EZVs9Hxz/automated-ds-ml).

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Installation](#installation)
- [Development Phases](#development-phases)
- [Feedback](#feedback)
- [Contributors](#contributors)
- [Sponsors](#sponsors)
- [Acknowledgments](#acknowledgments)
- [For Developers](#for-developers)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

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
  - Pre-trained models - BERT, GPT2, etc.
  - Statistical tests - Anova, T-test, etc.
  
You can view a full list of implemented techniques in the documentation or here: [TECHNIQUES.md](https://github.com/Ashton-Sidhu/aethos/blob/develop/TECHNIQUES.md)

Plus more coming soon such as:

  - Testing for model drift
  - Recommendation models
  - Parralelization through Dask and/or Spark
  - Uniform API for deep learning models
  - Automated code and file generation for jupyter notebook development and a python file of your data pipeline.

Aethos makes it easy to PoC, experiment and compare different techniques and models from various libraries. From imputations, visualizations, scaling, dimensionality reduction, feature engineering to modelling, model results and model deployment - all done with a single, human readable, line of code!

Aethos utilizes other open source libraries to help enhance your analysis from enhanced stastical information, interactive visual plots or statistical tests and models - all your tools in one place, all accessible with one line of code or a click! See below in the [Acknowledgments](#acknowledgments) for the open source libraries being used in this project.

## Usage
For full documentation on all the techniques and models, click [here](https://aethos.readthedocs.io/en/latest/?badge=latest) or [here](https://aethos.readthedocs.io/en/latest/source/aethos.html#)

Examples can be viewed [here](https://github.com/Ashton-Sidhu/aethos/tree/develop/examples)

To start, we need to import Aethos dependencies as well as pandas.

Before that, we can create a full data science folder structure by running `aethos create` from the command line and follow the command prompts.

### Options

To enable extensions, such as QGrid interactive filtering, enable them as you would in pandas:

```python
import aethos as at

at.options.interactive_df = True
```

Currently the following options are:

  - `interactive_df`: Interactive grid with QGrid
  - `interactive_table`: Interactive grid with Itable - comes with built in client side searching
  - `project_metrics`: Setting project metrics
    - Project metrics is a metric or set of metrics to evaluate models.
  - `track_experiments`: Uses MLFlow to track models and experiments.

User options such as changing the directory where images, and projects are saved can be edited in the config file. This is located at `USER_HOME`/.aethos/ .

This location is also the default location of where any images, and projects are stored.

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

Now all modelling and anlysis can be achieved via one object.


### Analysis

```python
import aethos as at
import pandas as pd

x_train = pd.read_csv('https://raw.githubusercontent.com/Ashton-Sidhu/aethos/develop/examples/data/train.csv') # load data into pandas

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

df.column_info() # Display info about each column in your data

df.describe_column('Fare') # Get indepth statistics about the 'Fare' column

df.mean() # Run pandas functions on the aethos objects

df.missing_data # View your missing data at anytime

df.correlation_matrix() # Generate a correlation matrix for your training data

df.predictive_power() # Calculates the predictive power of each variable

df.autoviz() # Runs autoviz on the data and runs EDA on your data

df.pairplot() # Generate pairplots for your training data features at any time

df.checklist() # Will provide an iteractive checklist to keep track of your cleaning tasks
```

**NOTE:** One of the benefits of using `aethos` is that any method you apply on your train set, gets applied to your test dataset. For any method that requires fitting (replacing missing data with mean), the method is fit on the training data and then applied to the testing data to avoid data leakage.

```python
# Replace missing values in the 'Fare' and 'Embarked' column with the most common values in each of the respective columns.
df.replace_missing_mostcommon('Fare', 'Embarked')

# To create a "checkpoint" of your data (i.e. if you just want to test this analytical method), assign it to a variable
df.replace_missing_mostcommon('Fare', 'Embarked')

# Replace missing values in the 'Age' column with a random value that follows the probability distribution of the 'Age' column in the training set. 
df.replace_missing_random_discrete('Age')

df.drop('Cabin') # Drop the cabin column
```

As you've started to notice, alot of tasks to df the data and to explore the data have been reduced down to one command, and are also customizable by providing the respective keyword arguments (see documentation).


```python
# Create a barplot of the mean surivial rate grouped by age.
df.barplot(x='Age', y='Survived', method='mean')

# Plots a scatter plot of Age vs. Fare and colours the dots based off the Survived column.
df.scatterplot(x='Age', y='Fare', color='Survived')

# One hot encode the `Person` and `Embarked` columns and then drop the original columns
df.onehot_encode('Person', 'Embarked', drop_col=True) 
```

### Modelling

#### Running a Single Model

Models can be trained one at a time or multiple at a time. They can also be trained by passing in the params for the sklearn, xgboost, etc constructor, by passing in a gridsearch dictionary & params, cross validating with gridsearch & params.

After a model has been ran, it comes with use cases such as plotting RoC curves, calculating performance metrics, confusion matrices, SHAP plots, decision tree plots and other local and global model interpretability use cases.

```python
lr_model = df.LogisticRegression(random_state=42) # Train a logistic regression model

# Train a logistic regression model with gridsearch
lr_model = df.LogisticRegression(gridsearch={'penalty': ['l1', 'l2']}, random_state=42)

# Crossvalidate a a logistic regression model, displays the scores and the learning curve and builds the model
lr_model = df.LogisticRegression()
lr_model.cross_validate(n_splits=10) # default is strat-kfold for classification  problems

# Build a Logistic Regression model with Gridsearch and then cross validates the best model using stratified K-Fold cross validation.
lr_model = model.LogisticRegression(gridsearch={'penalty': ['l1', 'l2']}, cv_type="strat-kfold") 

lr_model.help_debug() # Interface with items to check for to help debug your model.

lr_model.metrics() # Views all metrics for the model
lr_model.confusion_matrix()
lr_model.decision_boundary()
lr_model.roc_curve()
```

#### Running multiple models in parallel

```python
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
```

#### Using Pretrained Models

Currently you can use pretrained models such as BERT, XLNet, AlBERT, etc. to calculate sentiment and answer questions.

```python
df.pretrained_sentiment_analysis(`text_column`)

# To answer questions, context for the question has to be supplied
df.pretrained_question_answer(`context_column`, `question_column`)
```

### Model Interpretability

As mentioned in the Model section, whenever a model is trained you have access to use cases for model interpretability as well. There are prebuild SHAP usecases and an interactive dashboard that is equipped with LIME and SHAP for local model interpretability and Morris Sensitivity for global model interpretability.

```python
lr_model = model.LogisticRegression(random_state=42)

lr_model.summary_plot() # SHAP summary plot
lr_model.force_plot() # SHAP force plot
lr_model.decision_plot() # SHAP decision plot
lr_model.dependence_plot() # SHAP depencence plot

# Creates an interactive dashboard to interpret predictions of the model
lr_model.interpret_model() 
```

### Code Generation

Currently you are only able to export your model to be ran a service, and will be able to automatically generate the required files. The automatic creation of a data pipeline is still in progress.

```python

lr_model.to_service('titanic')
```

Now navigate to 'your_home_folder'('~' on linux and Users/'your_user_name' on windows)/.aethos/projects/titanic/ and you will see the files needed to run the model as a service using FastAPI and uvicorn. 

## Installation

**Python Requirements**: 3.6, 3.7

`pip install aethos`

To install the dependencies to use pretrained models such as BERT, XLNet, AlBERT, etc:

`pip install aethos[ptmodels]`

To install associating corpora for nltk analysis:

`aethos install-corpora`

To install and use the extensions such as `qgrid` for interactive filtering and analysis with DataFrames:

`aethos enable-extensions`

Currently working on condas implementation.

To create a Data Science project run:

`aethos create`

This will create a full folder strucuture for you to manage data, unit tests, experiments and source code.

If experiment tracking is enabled or if you want to start the MLFlow UI:

`aethos mlflow-ui`

This will start the MLFlow UI in the directory where your Aethos experiemnts are run.
NOTE: This only works for local use of MLFLOW, if you are running MLFlow on a remote server, just start it on the remote server and enter the address in the `%HOME%/.aethos/config.yml` file.


## Development Phases

#### Phase 1
  - [x]	Data Processing techniques
    - [x] Data Cleaning V1
    - [x] Feature Engineering V1
  - [x]	Reporting V1

#### Phase 2
  - [x]	Data visualizations
  - [x]	Models and Evaluation
  - [x]	Reporting V2

#### Phase 3
  - [x] Quality of life/addons
  - [x] Code Generation V1
  - [x] Experiment Tracking
  - [x] Pre trained models

#### Phase 4
  - [ ]	Deep learning integration
  - [x] Statistical Tests
  - [ ] Recommendation Models
  - [ ] Code Generation V2
    
#### Phase 5
  - [ ] Add time series models (i.e ARIMA) and feature engineering
  - [ ] Parallelization (Spark, Dask, etc.)
  - [ ]	Cloud computing
  - [ ] Graph based learning and representation

#### Phase 6
  - [ ] Web App
  
These are subject to change.

## Feedback

I appreciate any feedback so if you have any feature requests or issues make an issue with the appropriate tag or futhermore, send me an email at sidhuashton@gmail.com

## Contributors

This project follows the [all-contributors](https://github.com/kentcdodds/all-contributors) specification and is brought to you by these [awesome contributors](./CONTRIBUTORS.md).

## Sponsors

N/A

## Acknowledgments

Credits go to the backbone of open source DataScience and ML: Pandas, Numpy, Scipy, Scikit Learn, Matplotlib, Plotly, Gensim and Jupyter.

Community credits go to:

[@mouradmourafiq](https://github.com/mouradmourafiq) for his [pandas-summary](https://github.com/mouradmourafiq/pandas-summary) library.

[@PatrikHlobil](https://github.com/PatrikHlobil) for his [Pandas-Bokeh](https://github.com/PatrikHlobil/Pandas-Bokeh) library.

[@pandas-profiling](https://github.com/pandas-profiling) for their automated [EDA report generation](https://github.com/pandas-profiling/pandas-profiling) library.

[@slundberg](https://github.com/slundberg/) for his [shap](https://github.com/slundberg/shap) model explanation library.

[@microsoft](https://github.com/microsoft/) for their [interpret](https://github.com/microsoft/interpret) model explanation library.

[@DistrictDataLabs](https://github.com/DistrictDataLabs?type=source) for their [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) visual analysis and model diagnostic tool.

[@Quantopian](https://github.com/quantopian?type=source) for their interactive DataFrame library [qgrid](https://github.com/quantopian/qgrid).

[@mwouts](https://github.com/mwouts) for their interactive Dataframe library [itable](https://github.com/mwouts/itables).

[@jmcarpenter2](https://github.com/jmcarpenter2/) for his parallelization library [Swifter](https://github.com/jmcarpenter2/swifter).

[@mlflow](https://github.com/mlflow/) for their model tracking library [mlflow](https://github.com/mlflow/mlflow/).

[@huggingface](https://github.com/huggingface/) for their automated pretrained model library [transformers](https://github.com/huggingface/transformers).

[@AutoViML](https://github.com/AutoViML/) for their auto visualization library [autoviz](https://github.com/AutoViML/AutoViz).

[@8080labs](https://github.com/8080labs/) for their predictive power score library [ppscore](https://github.com/8080labs/ppscore).

## For Developers

To install packages `pip3 install -r requirements-dev.txt`

To run tests `python3 -m unittest discover aethos/`
