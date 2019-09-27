
*"A collection of tools for Data Scientists and ML Engineers for them to focus less on how to do the analysis and instead worry about what are the best analytic tools that will help gain the most insights from their data."*

Welcome to PyAutoMl's documentation!
====================================
Py-automl is a library/platform that automates your data science and analytical tasks at any stage in the pipeline. Py-automl is, at its core, a wrapper that helps automate analytical techniques from various libaries such as pandas, sci-kit learn, gensim, etc. Py-automl makes it easy to PoC, experiment and compare different techniques and models from various libraries. From cleaning your data, visualizing it and even applying feature engineering techniques from your favourite libraries - all done with a single, human readable, line of code!

Not to mention this process becomes easier with the use of the webapp, where it's just one click (Coming soon). When you're all done with your analysis, pick the techniques that best suit your data and with one line of code (or one click) get the python code of your pipeline thats transform your data to its processed state. You can also choose to just get the file of your transformed data and/or the model itself (Coming soon)!

Py-automl utilizes other open source libraries to help enhance your analysis from enhanced stastical information, interactive visual plots or statistical tests and models - all in one place, all accessible with one line of code or a click!

For more info such as features, development plan, status and vision checkout the `Py-automl github page <https://github.com/Ashton-Sidhu/py-automl/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Usage
=====

To start, we need to import the data science workflow stages as well as pandas.

.. code:: python

    from pyautoml import Clean, Preprocess, Feature
    import pandas as pd

Load your data into pandas.

.. code:: python

    x_train = pd.read_csv('data/train.csv')

So as is almost always the case, let's start with cleaning the data. We load our data now into the cleaning phase.

.. code:: python

    clean = Clean(data=x_train, target_field='Survived', report_name='Titanic')

Couple of key points to note, ``Clean`` takes in quite a few keyword arguments. Other than ``data``, if your data has already been pre-split into training and testing data, you can start the cleaning phase like this:

.. code:: python

    clean = Clean(x_train=x_train, x_test=x_test, split=False, target_field='Survived', report_name='Titanic')

Since our data is not presplit, we use the ``data`` parameter to indicate that and as a result our data is automatically split to avoid data leakage. Note that the default split percentage is ``20%`` but is configurable by using the keyword argument ``test_split_percentage``.

Target field is the field we are trying to predict in this case, if there is no field to predict there is no need to pass in that argument (note: ``Survived`` is the column name in our dataset). The last keyword argument used is ``report_name`` and that names our continuously updated and automated report that will be saved in your current working directory. Every technique you apply on your dataset will be logged along with some "reasoning" about why it was done.

**NOTE:** One of the benefits of using ``pyautoml`` is that any method you apply on your train set, gets applied to your test dataset. For any method that requires fitting (replacing missing data with mean), the method is fit on the training data and then applied to the testing data to avoid data leakage.

Now that our data has been loaded, there a few ways we can explore and gain initial insights from our data. To start, at any time and with **ANY** ``pyautoml`` object (Clean, Preprocess, Feature, etc) you can view your data with the following commands:

.. code:: python

    clean.data # If your data IS NOT split
    clean.x_train # If your data IS split
    clean.x_test # If your data IS split

Also you can view a glance of your full data (if it has not been split) or your training dataset at any time by just calling the object (like pandas):

.. code:: python

    clean # This will give you a glance of your full data or your training data, whichever is provided

Also note you can interface any of the ``pyautoml`` objects like pandas, for example if you want to filter by ``Age``:

.. code:: python

    clean[clean.Age > 25]

This will give you a look at your data where the record has a value of > 25 in the ``Age`` column.

You can also create columns how you would in pandas using either dot notation or brackets:

.. code:: python

    clean.new_col = YOUR_DATA_HERE
    clean['new_col'] = YOUR_DATA_HERE

**NOTE:** If you are providing a list or a Series and your data is split into train and test, the new column is created in the dataset that matches the length of the data provided. If the length of the data provided matches both train and test data it is added to both. To individually add new columns you can do the following:

.. code:: python

    clean.x_train['new_col'] = YOUR_DATA_HERE
    clean.x_test['new_col'] = YOUR_DATA_HERE

To get a full report of your data at anytime, you can run ``data_report`` and a full report will be generated from your data courtesy of the ``pandas-profiling`` library.

.. code:: python

    clean.data_report(title='Titanic Summary', output_file='titanic_summary.html')

To get a faster more high level view of your data, you can run ``describe()`` and get an extension of the pandas describe function.

.. code:: python

    clean.describe()

For indepth information and statistics about a column you can use the following:

.. code:: python

    clean.describe_column('Fare') # Fare is the name of the column

At any point you can get a summary of missing values in all your datasets by running the following:

.. code:: python

    clean.missing_data

Now to deal with missing data by replacing them with the most common value in that column, run the following command:

.. code:: python

    clean.replace_missing_mostcommon('Fare', 'Embarked')

This will replace the missing values in the ``Fare`` and ``Embarked`` columns with the most common values in each of its respective columns.

**NOTE:** In pandas you'll often see ``df = df.method(...)`` or ``df.method(..., inplace=True)`` when transforming your data. Then depending on how you developed your analysis, when a mistake is made you either have to restart the kernel or reload your data entirely. In ``pyautoml`` most methods will change the data inplace (methods that have the keyword argument ``new_col_name`` will create a new column) without having to go ``df = df.method(...)``. To create a "checkpoint" that creates a copy of your current state just assign the method to a variable, for example:

.. code:: python

    rep_mcommon = clean.replace_missing_mostcommon('Fare', 'Embarked')

Now I can keep going with my analysis using the clean object and if something goes wrong when exploring that analysis path, I can pick right up from this point by using the ``rep_mcommon`` variable, without having to restart any kernels or reload any data.

Another example, we can replace missing values in the ``Age`` column with a random value that follows the distribution of that column as follows:

.. code:: python

    clean.replace_missing_random_discrete('Age)

At any point we can drop columns as follows:

.. code:: python

    clean.drop('Cabin')

Columns can also be dropped by defining the columns you want to keep (drop all columns except the ones you want to keep) or by passing in a regex expressions and all columns that match the regex expression will be dropped.

As you've started to notice, alot of tasks to clean the data and to explore the data have been reduced down to one command, and are also customizable by providing the respective keyword arguments (see documentation).

Creating visualisations has never been easier to, for example viewing the mean survival rate based off the age of the passenger can be done as follows:

.. code:: python

    clean.visualize_barplot('Age', 'Survived', groupby='Age', method='mean', xlabel='Age')

This will create an interactive Bokeh plot that can be exported as an html to be embedded or viewed on a webpage or just as an image file.

Once you are done cleaning your data and are ready to move onto Preproccessing/Normalizing your data or Feature Engineering, it's as easy as doing the following:

.. code:: python

    prep = Preprocess(clean)

And now you can preprocess/normalize your data with the automated techniques such as normalizing numeric values between 0 and 1.

.. code:: python

    feature = Feature(clean)

Or you can now transform your data or abstract new features from your data through the feature class, keeping your workflow organized and easy to understand.

In terms of speed, on the backend I am doing everything I can do to use vectorization to reduce processing and computation time (even when using .apply) and I am constantly trying to make speed improvements where possible.

Installation
============

For package use (no GUI):

``pip install py-automl``

Currently working on condas implementation.

General Methods and Visualizations
==================================

For general methods and visualization docmentation click `here <https://py-automl.readthedocs.io/en/latest/source/pyautoml.html#module-pyautoml.base/>`_.

Cleaning Methods
================

For cleaning docmentation click `here <https://py-automl.readthedocs.io/en/latest/source/pyautoml.cleaning.html#module-contents/>`_.

Preprocessing Methods
=====================

For preprocessing docmentation click `here <https://py-automl.readthedocs.io/en/latest/source/pyautoml.preprocessing.html#module-pyautoml.preprocessing/>`_.

Feature Engineering Methods
===========================

For feature engineering docmentation click `here <https://py-automl.readthedocs.io/en/latest/source/pyautoml.feature_engineering.html#module-pyautoml.feature_engineering/>`_.

Modelling Methods
=================

For modelling docmentation click `here <https://py-automl.readthedocs.io/en/latest/source/pyautoml.modelling.html#module-pyautoml.modelling/>`_.

Examples
========

Examples can be viewed `here <https://github.com/Ashton-Sidhu/py-automl/tree/develop/examples/>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
