# Project Title

**Kedro-Powered Machine Learning Pipeline for Santander Customer Satisfaction Prediction**

## Overview

This project showcases a comprehensive machine learning pipeline for predicting customer satisfaction for Santander, based on the Kaggle competition "[Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction)." This pipeline is built using Kedro, an open-source data pipeline development framework, and is designed to provide end-to-end support for data engineering and machine learning tasks.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Why Kedro?](#why-kedro)
4. [Project Structure](#project-structure)
5. [Pipeline Overview](#pipeline-overview)
6. [Acknowledgments](#acknowledgments)

## Introduction

Predicting customer satisfaction is a crucial task for businesses in various industries. Santander, a global financial services provider, has initiated a Kaggle competition to predict whether a customer is satisfied or not based on a set of features.

This project uses the Kaggle competition as a real-world problem to demonstrate the power and efficiency of the Kedro framework. Kedro offers a structured and maintainable way to develop, test, and deploy data pipelines and machine learning models.

## Getting Started

To get started with this project, you'll need to:

1. Clone this repository to your local machine.
2. Install Kedro if you haven't already:
   
   ```bash
   pip install kedro
   ```
   
3. Create a Kedro project by running:
   
   ```bash
   kedro new
   ```
   
   Follow the prompts to set up your Kedro project.
   
4. Copy the content of this repository into your Kedro project's directory structure.

Now, you can run the Kedro pipeline to execute all the data preprocessing, feature selection, model training, and scoring tasks.

## Why Kedro?

Kedro is a powerful framework for data pipeline development that is well-suited for machine learning projects in software engineering. Here are some key reasons why Kedro is valuable for this project:

- **Data Engineering Best Practices:** Kedro encourages best practices in data engineering by promoting data cataloging, versioning, and testing.

- **Reproducibility:** With Kedro, you can ensure that all your experiments and models are fully reproducible.

- **Modularity:** Kedro allows you to structure your project in a modular way, separating tasks like feature engineering, model training, and model scoring into separate nodes.

- **Flexible Data Loading:** Kedro's data catalog makes it easy to load, save, and share data between nodes in the pipeline.

- **Data Versioning:** Kedro keeps track of data versions and helps you manage changes to your data as your project evolves.

## Project Structure

In this project, we have organized our code into different directories to maintain a structured approach:

- **src/feature_selection:** Contains code for unsupervised and supervised feature selection.
- **src/model_training:** Includes code for training a machine learning model.
- **src/model_scoring:** Contains code for evaluating the model and scoring new data.
- **src/pre_process:** Includes code for data preprocessing, such as train-test splitting.

## Pipeline Overview

Our Kedro pipeline is designed to perform the following tasks:

1. **Data Preprocessing:** We perform data preprocessing and train-test splitting to prepare the data for feature selection and model training.

2. **Feature Selection:** This stage includes unsupervised and supervised feature selection to identify the most relevant features for model training.

3. **Model Training:** We use the selected features to train a machine learning model, in this case, an XGBoost classifier.

4. **Model Evaluation:** We evaluate the trained model's performance using metrics such as ROC AUC.

5. **Model Scoring:** Finally, we use the trained model to score new data and generate deciles for Santander customer satisfaction predictions.

By structuring our project and pipeline this way, we can easily test and iterate on different feature selection techniques and models while keeping our code organized.
