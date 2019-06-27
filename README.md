# py-automl

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Development Timeline](#development-timeline)
- [Feedback](#feedback)
- [Contributors](#contributors)
- [Sponsors](#sponsors)
- [Acknowledgments](#acknowledgments)
- [Developers](#for-developers)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

Py-automl is a platform (Web Application and API) that automates tasks of the ML pipeline, from Data Cleaning to Validation. Py-automl is designed to aggregate machine learning techniques and models so that they are usable by everyone from Data Scientists and Machine Learning Engineers to Data Analysts to Business Professionals. It gives users full customizability and visibility into the techniques that are being used yet also comes with an autoML feature (soon). Each technique has customizable parameters where applicable that can be passed in as arguments to help automate tasks. Every part of the auto-ml pipeline with automated and users can start automating at any point (i.e. if the user already has cleaned their dataset, they can start automating from the feature engineering/extraction phase).

Py-automl provides you with the code for each technique that was ran on your dataset to remove the "black box" of other autoML platforms. This allows users to learn, customize and tweak the code as they desire. The code provided will be production-ready so you don't have to waste time writing the code and then revising it to production standard. If any models were ran, the users will receive the trained models. As py-automl goes through the ML pipeline it records its actions and steps provides a detailed report of what was done, how it was done, where it was done, etc. allowing users to share their process with co-workers, colleagues, friends, etc.

It is Py-automls's goal that Data Scientists and Machine Learning Engineers will contribute the techniques they have used and that researchers will contribute with their code and paper so that everyone using the platform can apply the latest advancements and techniques in A.I. onto their dataset.

## Features

## Development Timeline
#### Phase 1
  - [ ]	UI framework
  - [ ]	Data Processing techniques
    - [x] Data Cleaning V1
    - [ ] Feature Engineering V1
  - [ ]	Code Generation
  - [ ]	Reporting V1

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

These are subject to change.

## Feedback

I appreciate any feedback so if you have any feature requests or issues make an issue with the appropriate tag or futhermore, send me an email at ashton.sidhu1994@gmail.com

## Contributors

This project follows the [all-contributors](https://github.com/kentcdodds/all-contributors) specification and is brought to you by these [awesome contributors](./CONTRIBUTORS.md).

## Build Process

## Sponsors

## Acknowledgments

## For Developers

To generate requirements file install `pipreqs`, `pip install pipreqs`.

Navigate to py-automl root directory and run `pipreqs .`
