[![PyPI version](https://badge.fury.io/py/aethos.svg)](https://badge.fury.io/py/aethos) [![CircleCI](https://circleci.com/gh/Ashton-Sidhu/aethos/tree/develop.svg?style=svg)](https://circleci.com/gh/Ashton-Sidhu/aethos/tree/develop) [![Documentation Status](https://readthedocs.org/projects/aethos/badge/?version=latest)](https://aethos.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/Ashton-Sidhu/aethos/branch/develop/graph/badge.svg)](https://codecov.io/gh/Ashton-Sidhu/aethos)



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
  - Automated reporting - as you perform your analysis, a report is created along side with it
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

Plus more coming soon such as:

  - Testing for model drift
  - Recommendation models
  - Statistical tests - Anova, T-test, etc.
  - Pre-trained models - BERT, GPT2, etc.
  - Parralelization through Dask and/or Spark
  - Uniform API for deep learning models and automated code and file generation jupyter notebook development, python file of your data    pipeline.

Aethos makes it easy to PoC, experiment and compare different techniques and models from various libraries. From imputations, visualizations, scaling, dimensionality reduction, feature engineering to modelling, model results and model deployment - all done with a single, human readable, line of code!

Aethos utilizes other open source libraries to help enhance your analysis from enhanced stastical information, interactive visual plots or statistical tests and models - all your tools in one place, all accessible with one line of code or a click! See below in the [Acknowledgments](#acknowledgments) for the open source libraries being used in this project.

## Usage
For full documentation on all the techniques and models, click [here](https://aethos.readthedocs.io/en/latest/?badge=latest) or [here](https://aethos.readthedocs.io/en/latest/source/aethos.html#)

Examples can be viewed [here](https://github.com/Ashton-Sidhu/aethos/tree/develop/examples)

To start, we need to import the ethos dependencies as well as pandas.

Before that, we can create a full data science folder structure by running `aethos create` from the command line and follow the command prompts.

For a full list of methods please see the full docs or [TECHNIQUES.md]()

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
  - `word_doc`: Writes report to a word document as well as the .txt file

User options such as changing the directory where images, reports, and projects are saved can be edited in the config file. This is located at `USER_HOME`/.aethos/ .

This location is also the default location of where any images, reports and projects are stored.

### Analysis

```python
import aethos as at
import pandas as pd

x_train = pd.read_csv('data/train.csv') # load data into pandas

# Initialize Data object with training data
# By default, if no test data (x_test) is provided, then the data is split with 20% going to the test set
# Specify predictor field as 'Survived'
# Specify report name
df = at.Data(x_train, target_field='Survived', report_name='Titanic')

df.x_train # View your training data
df.x_test # View your testing data

df # Glance at your training data

df[df.Age > 25] # Filter the data

df['new_col'] = [1, 2]  # Add a new column to the data, based off the length of the data provided, it will add it to the train or test set.

df.x_train['new_col'] = [1,2] # This is the exact same as the either of code above
df.x_test['new_col'] = [1,2]

df.data_report(title='Titanic Summary', output_file='titanic_summary.html') # Automate EDA with pandas profiling with an autogenerated report

df.describe() # Display a high level view of your data using an extended version of pandas describe

df.describe_column('Fare') # Get indepth statistics about the 'Fare' column

df.mean() # Run pandas functions on the aethos objects

df.missing_data # View your missing data at anytime

df.correlation_matrix() # Generate a correlation matrix for your training data

df.pairplot() # Generate pairplots for your training data features at any time

df.checklist() # Will provide an iteractive checklist to keep track of your cleaning tasks
```

**NOTE:** One of the benefits of using `aethos` is that any method you apply on your train set, gets applied to your test dataset. For any method that requires fitting (replacing missing data with mean), the method is fit on the training data and then applied to the testing data to avoid data leakage.

```python
df.replace_missing_mostcommon('Fare', 'Embarked') # Replace missing values in the 'Fare' and 'Embarked' column with the most common values in each of the respective columns.

df.replace_missing_mostcommon('Fare', 'Embarked') # To create a "checkpoint" of your data (i.e. if you just want to test this analytical method), assign it to a variable

df.replace_missing_random_discrete('Age') # Replace missing values in the 'Age' column with a random value that follows the probability distribution of the 'Age' column in the training set. 

df.drop('Cabin') # Drop the cabin column
```

As you've started to notice, alot of tasks to df the data and to explore the data have been reduced down to one command, and are also customizable by providing the respective keyword arguments (see documentation).


```python
df.barplot(x='Age', y=['Survived'], method='mean', xlabel='Age') # Create a barblot of the mean surivial rate grouped by age.

df.onehot_encode('Person', 'Embarked', drop_col=True) # One hot encode the `Person` and `Embarked` columns and then drop the original columns
```

### Modelling

```python
model = at.Model(df)
```

#### Running a Single Model

Models can be trained one at a time or multiple at a time. They can also be trained by passing in the params for the sklearn, xgboost, etc constructor, by passing in a gridsearch dictionary & params, cross validating with gridsearch & params.

After a model has been ran, it comes with use cases such as plotting RoC curves, calculating performance metrics, confusion matrices, SHAP plots, decision tree plots and other local and global model interpretability use cases.

```python
lr_model = model.LogisticRegression(random_state=42) # Train a logistic regression model
lr_model = model.LogisticRegression(gridsearch={'penalty': ['l1', 'l2']}, random_state=42) # Trains a logistic regression model with gridsearch
lr_model = model.LogisticRegression(cv=5, n_splits=10) # Crossvalidates a logistic regression model, displays the scores and the learning curve and builds the model
lr_model = model.LogisticRegression(gridsearch={'penalty': ['l1', 'l2']}, cv='strat-kfold', n_splits=10) # Builds a Logistic Regression model with Gridsearch and then cross validates the best model using stratified K-Fold cross validation.

lr_model.metrics() # Views all metrics for the model
lr_model.confusion_matrix()
lr_model.roc_curve()
```

#### Running multiple models in parallel

```python
model.LogisticRegression(random_state=42, model_name='log_reg', run=False) # Adds a logistic regression model to the queue
model.RandomForestClassification(run=False) # Adds a random forest model to the queue
model.XGBoostClassification(run=False) # Adds an xgboost classification model to the queue

model.run_models() # This will run all queued models in parallel
model.run_models(method='series') # Run each model one after the other

model.compare_models() # This will display each model evaluated against every metric

# Every model is accessed by a unique name that is assiged when you run the model.
# Default model names can be seen in the function header of each model.

model.log_reg.confusion_matrix() # Displays a confusion matrix for the logistic regression model
model.rf_cls.confusion_matrix() # Displays a confusion matrix for the random forest model
```

### Model Interpretability

As mentioned in the Model section, whenever a model is trained you have access to use cases for model interpretability as well. There are prebuild SHAP usecases and an interactive dashboard that is equipped with LIME and SHAP for local model interpretability and Morris Sensitivity for global model interpretability.

```python
lr_model = model.LogisticRegression(random_state=42)

lr_model.summary_plot() # SHAP summary plot
lr_model.force_plot() # SHAP force plot
lr_model.decision_plot() # SHAP decision plot
lr_model.dependence_plot() # SHAP depencence plot

lr_model.interpret_model() # Creates an interactive dashboard to view LIME, SHAP, Morris Sensitivity and more for your model
```

### Code Generation

Currently you are only able to export your model to be ran a service, and will be able to automatically generate the required files. The automatic creation of a data pipeline is still in progress.

```python

lr_model.to_service('titanic')
```

Now navigate to 'your_home_folder'('~' on linux and Users/'your_user_name' on windows)/.aethos/projects/titanic/ and you will see the files needed to run the model as a service using FastAPI and uvicorn. 

## Installation

`pip install aethos`

To install associating corpora for nltk analysis:

`aethos install-corpora`

To install and use the extensions such as `qgrid` for interactive filtering and analysis with DataFrames:

`aethos enable-extensions`

Currently working on condas implementation.

To create a Data Science project run:

`aethos create`

This will create a full folder strucuture for you to manage data, unit tests, experiments and source code.

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
  - [ ] Pre trained models

#### Phase 4
  - [ ]	Deep learning integration
  - [ ] Statistical Models
  - [ ] Recommendation Models
  - [ ] Code Generation V2
    
#### Phase 5
  - [ ] Parallelization (Spark, Dask, etc.)
  - [ ]	Cloud computing
  - [ ] Graph based learning and representation

#### Phase 6
  - [ ] Web App
  
These are subject to change.

## Feedback

I appreciate any feedback so if you have any feature requests or issues make an issue with the appropriate tag or futhermore, send me an email at ashton.sidhu1994@gmail.com

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

[@mwouts](https://github.com/mwouts) for their interactive Dataframe library [itable](https://github.com/mwouts/itables)

## For Developers

For python snippets to make deving new techniques either, message me and I can send them.

### Contributing data analysis techniques

  1. The code for the transformation belongs in the `numeric`, `categorical`, etc. file for the stage it belongs.
  2. In the stage.py (`clean.py`, `preprocess.py`, etc. ) file, call the analytical method as well as add the reporting code
  3. Add a description for the technique in the `aethos/config/technique_reasons.yml` file.
  4. Write the unit test in the `test_unittest.py` file in the stage folder.

### Contributing models

  1. Define the model as its own function in `model.py`
  2. The import statements happen within the function.
  3. Call the approriate `_run_` function (supervised or unsupervised)
      - Text models go in text.py and call them from model.py
  4. Add the model type to the dictionaries `SHAP_LEARNER` AND `PROBLEM_TYPE` dict in `model/constants.py`
  5. Write the unittest in `test_unittest.py`

### Contributing model analysis use cases

  1. Generic model analysis goes in `model_analysis.py`
  2. Model interpretability goes in `model_explanation.py`

To install packages `pip3 install -r requirements.txt`

To run tests `python3 -m unittest discover aethos/`
