[![PyPI version](https://badge.fury.io/py/py-automl.svg)](https://badge.fury.io/py/py-automl) [![CircleCI](https://circleci.com/gh/Ashton-Sidhu/py-automl/tree/develop.svg?style=svg)](https://circleci.com/gh/Ashton-Sidhu/py-automl/tree/develop) [![Documentation Status](https://readthedocs.org/projects/py-automl/badge/?version=latest)](https://py-automl.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/Ashton-Sidhu/py-automl/branch/develop/graph/badge.svg)](https://codecov.io/gh/Ashton-Sidhu/py-automl)




# py-automl

<i>"A collection of tools for Data Scientists and ML Engineers for them to focus less on how to do the analysis and instead worry about what are the best analytic tools that will help gain the most insights from their data."</i>

To track development of the project, you can view the [Trello board](https://trello.com/b/EZVs9Hxz/automated-ds-ml).

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)
- [Usage](#usage)
- [Installation](#installation)
- [Features](#features)
- [Development Phases](#development-phases)
- [Feedback](#feedback)
- [Contributors](#contributors)
- [Sponsors](#sponsors)
- [Acknowledgments](#acknowledgments)
- [For Developers](#for-developers)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

Py-automl is a library/platform that automates your data science and analytical tasks at any stage in the pipeline. Py-automl is, at its core, a wrapper that helps automate analytical techniques from various libaries such as pandas, sci-kit learn, gensim, etc. and tries to the bridge the gap 

Py-automl makes it easy to PoC, experiment and compare different techniques and models from various libraries. From cleaning your data, visualizing it and even applying feature engineering techniques from your favourite libraries - all done with a single, human readable, line of code!

Py-automl utilizes other open source libraries to help enhance your analysis from enhanced stastical information, interactive visual plots or statistical tests and models - all your tools in one place, all accessible with one line of code or a click! See below in the [Acknowledgments](#acknowledgments) for the open source libraries being used in this project.

## Motivation

I created this library to help automate the data science/machine learning pipeline and have all the tools required for analysis in one place. I grew tired having to go back and look at code to find implementation for a certain type of analysis, googling for the implementation and converting PoCs to production level code or a microservice. I wanted to be able to focus more time on thinking about the analysis and the techniques to apply instead of worrying about developing them and finding implementations.

Secondly there were alot of people that wanted to try data science that were in a different technical stream but and were blocked due to knowledge and technical barriers. My goal was to remove the technical barriers, so that as long as they understand the techniques at a high level, they can work with data scientists and help contribute to performing some analysis with just one line of code or a click of a button - allowing the data scientist or ML engineer to focus on interpreting/applying the results.

For more info see my [vision statememt](https://github.com/Ashton-Sidhu/py-automl/blob/develop/VISION.md).

## Usage
For full documentation on all the techniques and models, click [here](https://py-automl.readthedocs.io/en/latest/?badge=latest) or [here](https://py-automl.readthedocs.io/en/latest/source/pyautoml.html#)

Examples can be viewed [here](https://github.com/Ashton-Sidhu/py-automl/tree/develop/examples)

To start, we need to import the data science workflow stages as well as pandas.

Before that, we can create a full data science folder structure by running `pyautoml create` from the command line and follow the command prompts.

#### General Use

```python
from pyautoml import Clean, Preprocess, Feature
import pandas as pd

x_train = pd.read_csv('data/train.csv') # load data into pandas

# Initialize cleaning object with training data
# By default, if no test data (x_test) is provided, then the data is split with 20% going to the test set
# Specify predictor field as 'Survived'
# Specify report name
clean = Clean(x_train=x_train, target_field='Survived', report_name='Titanic')

clean.x_train # View your training data
clean.x_test # View your testing data

clean # Glance at your training data

clean[clean.Age > 25] # Filter the data

clean.new_col = [1, 2] # Add a new column to the data, based off the length of the data provided, it will add it to the train or test set.
clean['new_col'] = [1, 2] # Another way

clean.x_train['new_col'] = [1,2] # This is the exact same as the either of code above
clean.x_test['new_col'] = [1,2]

clean.data_report(title='Titanic Summary', output_file='titanic_summary.html') # Automate EDA with pandas profiling with an autogenerated report

clean.describe() # Display a high level view of your data using an extended version of pandas describe

clean.describe_column('Fare') # Get indepth statistics about the 'Fare' column

clean.missing_data # View your missing data at anytime

clean.checklist() # Will provide an iteractive checklist to keep track of your cleaning tasks
```

#### Cleaning 

```python
clean.replace_missing_mostcommon('Fare', 'Embarked') # Replace missing values in the 'Fare' and 'Embarked' column with the most common values in each of the respective columns.

rep_mcommon = clean.replace_missing_mostcommon('Fare', 'Embarked') # To create a "checkpoint" of your data (i.e. if you just want to test this analytical method), assign it to a variable

# Now I can keep going with my analysis using the clean object and if something goes wrong when exploring this analysis path, I can pick right up from this point by using the `rep_mcommon` variable, without having to restart any kernels or reload any data.

clean.replace_missing_random_discrete('Age') # Replace missing values in the 'Age' column with a random value that follows the probability distribution of the 'Age' column in the training set. 

clean.drop('Cabin') # Drop the cabin column

# Columns can also be dropped by defining the columns you want to keep (drop all columns except the ones you want to keep) or by passing in a regex expressions and all columns that match the regex expression will be dropped.

# As you've started to notice, alot of tasks to clean the data and to explore the data have been reduced down to one command, and are also customizable by providing the respective keyword arguments (see documentation).
```

#### Preprocessing and Feature Engineering

```python
clean.visualize_barplot('Age', 'Survived', groupby='Age', method='mean', xlabel='Age') # Create a barblot of the mean surivial rate grouped by age.

prep = Preprocess(clean) # To move onto preprocessing

feature = Feature(clean) # to move onto feature engineering

feature.onehot_encode('Person', 'Embarked', drop_col=True) # One hot encode these columns and then drop the original columns
```

#### Modelling

```python
model = Model(feature) # To move onto modelling

# Models can be run in various ways

model.logistic_regression(random_state=42, run=True) # Train a logistic regression model
model.logistic_regression(gridsearch={'penalty': ['l1', 'l2']}, random_state=42, run=True) # Running gridsearch with the best params

model.logistic_regression(cv=5, learning_curve=True) # Crossvalidates a logistic regression model and displays the scores and the learning curve

model.logistic_regression(random_state=42, model_name='log_reg') # Adds a logistic regression model to the queue
model.random_forest() # Adds a random forest model to the queue
model.xgboost_classification() # Adds an xgboost classification model to the queue

model.run_models() # This will run all queued models in parallel
model.run_models(method='series') # Run each model one after the other

model.compare_models() # This will display each model evaluated against every metric

# Every model is accessed by a unique name that is assiged when you run the model.
# Default model names can be seen in the function header of each model.

model.log_reg.confusion_matrix() # Displays a confusion matrix for the logistic regression model

model.rf_cls.confusion_matrix() # Displays a confusion matrix for the random forest model
```

**NOTE:** One of the benefits of using `pyautoml` is that any method you apply on your train set, gets applied to your test dataset. For any method that requires fitting (replacing missing data with mean), the method is fit on the training data and then applied to the testing data to avoid data leakage.

**NOTE:** If you are providing a list or a Series and your data is split into train and test, the new column is created in the dataset that matches the length of the data provided. If the length of the data provided matches both train and test data it is added to both. To individually add new columns you can do the following:

**NOTE:** In pandas you'll often see `df = df.method(...)` or `df.method(..., inplace=True)` when transforming your data. Then depending on how you developed your analysis, when a mistake is made you either have to restart the kernel or reload your data entirely. In `pyautoml` most methods will change the data inplace (methods that have the keyword argument `new_col_name` will create a new column) without having to go `df = df.method(...)`. To create a "checkpoint" that creates a copy of your current state just assign the method to a variable, for example:

## Installation

`pip install py-automl`

To install associating corpora for nltk analysis:

`pyautoml -ic` or `pyautoml --install-corpora`

To install and use the extensions such as `qgrid` for interactive filtering and analysis with DataFrames:

`pyautoml -ie` or `pyautoml --install-extensions`

Currently working on condas implementation.

## Features

- Python package that simplifies and automates cleaning, visualizing, preprocessing, feature engineering, and modelling techniques.
- Report generation detailing exact steps how you transformed your dataset
- If you are doing a PoC or experimenting the code will output in a `.ipynb` and a `.py` format. *
- If the plan is to create a full pipeline the code will out a `.py` containing the full pipeline. *
- Model Evaluation
- Model Deployment *
- Spark Integration *
- Data visualizations
- On prem deployment *
- 3rd Party application integration (Azure, AWS, GC) *

## Development Phases

### Library
#### Phase 1
  - [x]	Data Processing techniques
    - [x] Data Cleaning V1
    - [x] Feature Engineering V1
  - [x]	Reporting V1

#### Phase 2
  - [x]	Data visualizations
  - [x]	Models and Evaluation
  - [ ]	Reporting V2

#### Phase 3
  - [ ] Quality of life/addons - IPR
  - [ ] Parallelization

#### Phase 4
  - [ ]	Spark
  - [ ]	Community centric optimization (making it easier to share techniques and models with other engineers).
    
#### Phase 5
  - [ ]	Cloud computing
  - [ ]	Deep learning integration

#### Phase 6
  - [ ] Web App
  - [ ] Code Generation
  
These are subject to change.

## Feedback

I appreciate any feedback so if you have any feature requests or issues make an issue with the appropriate tag or futhermore, send me an email at ashton.sidhu1994@gmail.com

## Contributors

This project follows the [all-contributors](https://github.com/kentcdodds/all-contributors) specification and is brought to you by these [awesome contributors](./CONTRIBUTORS.md).

## Sponsors

N/A

## Acknowledgments

[@mouradmourafiq](https://github.com/mouradmourafiq) for his [pandas-summary](https://github.com/mouradmourafiq/pandas-summary) library.

[@PatrikHlobil](https://github.com/PatrikHlobil) for his [Pandas-Bokeh](https://github.com/PatrikHlobil/Pandas-Bokeh) library.

[@pandas-profiling](https://github.com/pandas-profiling) for their automated [EDA report generation](https://github.com/pandas-profiling/pandas-profiling) library.

[@slundberg](https://github.com/slundberg/) for his [shap](https://github.com/slundberg/shap) model explanation library.

[@microsoft](https://github.com/microsoft/) for their [interpret](https://github.com/microsoft/interpret) model explanation library.

[@DistrictDataLabs](https://github.com/DistrictDataLabs?type=source) for their [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) visual analysis and model diagnostic tool.

## For Developers

For python snippets to make deving new techniques either, message me and I can send them.

To install packages `pip3 install -r requirements.txt`

To run tests `python3 -m unittest discover pyautoml/`
