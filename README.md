[![CircleCI](https://circleci.com/gh/Ashton-Sidhu/py-automl/tree/develop.svg?style=svg)](https://circleci.com/gh/Ashton-Sidhu/py-automl/tree/develop) [![Documentation Status](https://readthedocs.org/projects/py-automl/badge/?version=latest)](https://py-automl.readthedocs.io/en/latest/?badge=latest)

# py-automl

<i>"A collection of tools for Data Scientists and ML Engineers for them to focus less on how to do the analysis and instead worry about what are the best analytic tools that will help gain the most insights from their data."</i>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Development Phases](#development-phases)
- [Feedback](#feedback)
- [Contributors](#contributors)
- [Sponsors](#sponsors)
- [Acknowledgments](#acknowledgments)
- [Developers](#for-developers)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

Py-automl is a library/platform that automates your data science and analytical tasks at any stage in the pipeline. From cleaning your data, visualizing it and even applying feature engineering techniques from your favourite libraries - all done with a single, human readable, line of code! Not to mention this process becomes easier with the use of the webapp, where it's just one click (Coming soon). When you're all done with your analysis, pick the techniques that best suit your data and with one line of code (or one click) get the python code of your pipeline thats transform your data to its processed state. You can also choose to just get the file of your transformed data and/or the model itself (Coming soon)!

Py-automl utilizes other open source libraries to help enhance your analysis from enhanced stastical information, interactive visual plots or statistical tests and models - all in one place, all accessible with one line of code or a click! See below in the [Acknowledgments](#acknowledgments) for the open source libraries being used in this project.

## Motivation

I created this library to help automate the data science/machine learning pipeline and have all the tools required for analysis in one place. I hated having to go back and look at code to find implementation for a certain type of analysis or googling for the implementation. I wanted to be able to focus more time on thinking about the analysis and the techniques to apply instead of worrying about developing them and finding implementations. 

Secondly there were alot of people that wanted to do data science that were in a different technical stream but were blocked due to knowledge and technical barriers. My goal was to remove the technical barriers, so that as long as they understand the techniques at a high level, they can work with data scientists and help contribute to performing some analysis with just one line of code or a click of a button - allowing the data scientist or ML engineer to focus on interpreting/applying the results.

For more info see my [vision statememt](https://github.com/Ashton-Sidhu/py-automl/blob/develop/VISION.md).

## Features
- Python package that simplifies and automates cleaning, preprocessing, feature engineering, and modelling techniques.
- Web application that allows you to use those same packages through a GUI
- Report generation detailing exact steps how you transformed your dataset
- If automating workflow through the GUI, the application will generate the code that was ran on your data.
  - If you are doing a PoC or experimenting the code will output in a `.ipynb` and a `.py` format.
  - If the plan is to create a full pipeline the code will out a `.py` containing the full pipeline.
- Model Evaluation
- Spark Integration
- Data visualizations
- On prem deployment
- 3rd Party application integration (Azure, AWS, GC)

## Installation

For package use (no GUI): 

`pip install py-automl`

For Web App:

`In progress`

## Usage

Documentation can be viewed [here](https://py-automl.readthedocs.io/en/latest/py-modindex.html)

Proper usage documentation coming soon.

Examples can be viewed [here](https://github.com/Ashton-Sidhu/py-automl/tree/develop/examples)

## Development Phases

### Library
#### Phase 1
  - [x]	Data Processing techniques
    - [x] Data Cleaning V1
    - [x] Feature Engineering V1
  - [x]	Reporting V1

#### Phase 2
  - [ ]	Data visualizations
  - [ ]	Models and Evaluation
  - [ ]	Reporting V2

#### Phase 3
  - [ ] Parallelization

#### Phase 4
  - [ ]	Spark
  - [ ]	Community centric optimization (making it easier to share techniques and models with other engineers).
    
#### Phase 5
  - [ ]	Cloud computing
  - [ ]	Deep learning integration
  
### Web App

#### Phase 1
  - [x] Base Framework
  - [x] File Upload
  - [ ] Detect column type (categorical (numeric/string), numeric, string, text, etc.) - IPR
  - [ ] Display Data
  - [ ] Modify and Display data with technique choosing
  - [ ] Export final Result
  
### Code Generation

TBD

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

## For Developers

For python snippets to make deving new techniques either, message me and I can send them.

To install packages `pip3 install -r requirements.txt`

To run tests `python3 -m unittest discover pyautoml/`
