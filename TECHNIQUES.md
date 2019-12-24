Data Analysis Techniques
---

Note this does not include quality of life functions.

## Cleaning

  - Drop columns that have > x% of missing values
  - Drop columns of all constant values
  - Drop columns that have only unique values (id cols)
  - Drop rows that have > x% of missing values
  - Replace missing values with the mean of that column
  - Replace missing values with the median of that column
  - Replace missing values with the most common value of that column
  - Replace missing values with a specified constant
  - Replace missing values with a specific category
  - Remove rows if a value from a specific column is missing
  - Drop duplicate rows
  - Drop duplicate columns
  - Replace missing values with a random value that follows the distribution of that columns values
  - Replace missing values using KNN
  - Replace missing values with backfill
  - Replace missing values with forwardfill
  - Replace missing values with interpolation
  - Create a new column indicating if a value from that column is missing (Indicator column)

## Preprocessing

  - Normalize numeric data between 2 numbers using MaxMin
  - Normalize numeric data between quantiles
  - Normalize numeric data using log (natural, base 2 or base 10)
  - Split text data into its sentences
  - Stem text data
  - Splits text data into words
  - Remove stop words
  - Remove punctuation

## Feature Engineering

  - One hot encoding
  - TF-IDF
  - Bag of words
  - Text Hashing
  - PoS Tagging with spaCy
  - PoS Tagging with NLTK
  - Noun phrases with NLTK
  - Noun phrases with spaCy
  - Create polynomial features
  - Apply custom function (currently unoptimized)
  - Ordinally encode categorical variables
  - PCA Dim reduction
  - Drop correlated features

## Models

  - Extractive Text Summarization with TextRank
  - Extract Keywords with TextRank
  - Word2Vec
  - Doc2Vec
  - LDA
  - KMeans
  - DBScan
  - Isolation Forest
  - OneClass SVM
  - Agglomerative Clustering
  - Mean Shift
  - Gaussian Mixture Clustering
  - Logistic Regression
  - Ridge Classification + Regression
  - SGD Classification + Regression
  - ADABoost Classification + Regression
  - Bagging Classification + Regression
  - Gradient Boosting Classification + Regression
  - Random Forest Classification + Regression
  - Naive Bayes Bernoulli Classification
  - Naive Bayes Gaussian Classification
  - Naive Bayes Multinomial Classification
  - Decision Tree Classification + Regression
  - Linear SVC
  - SVC
  - XGBoost Classification + Regression
  - Linear Regression
  - Bayesian Ridge Regression
  - ElasticNet Regression
  - Lasso Regression
  - Linear SVR
  - SVR

## Model Analysis

  - View model weights and feature importance
  - SHAP Summary Plot
  - SHAP Decision Plot
  - SHAP Force Plot
  - SHAP Dependence Plot
  - Interpret Model with SHAP, LIME, Morris Sensitivity, etc.
  - Pickle Model
  - Generate code files to run model as a service
  - Filter data based on cluster
  - Plot clusters in 2d and 3d
  - View all classification and regression metrics, including sMAPE
  - Confusion Matrix
  - ROC Curve
  - Classification Report

## Visualizations
  - Raincloud plot
  - 2D and 3D Scatterplot
  - Barplot
  - Lineplot
  - Pairplot
  - Histograms
  - Jointplots
  - Correlation Matrix

## Statistical Methods
  - Predict whether sample is from train or test set to compare train and test set distributions
  - Kolomogirov Smirnov test for comparing feature distribution between train and test set
