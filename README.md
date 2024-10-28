# Breast Cancer Classification using Logistic Regression

This repository contains a project for classifying breast cancer cases as malignant or benign using a Logistic Regression model. The project focuses on analyzing various features of cell nuclei and building a classification model to predict the likelihood of malignancy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)

## Introduction

Breast Cancer Classification is an essential task for early diagnosis and effective treatment planning. This project aims to build a machine learning model to classify breast cancer cases based on different features of cell nuclei obtained from digitized images.

## Dataset

The dataset used in this project is the Breast Cancer dataset from the `sklearn.datasets` module. It contains various features of cell nuclei with corresponding labels indicating whether the cancer is benign or malignant. The dataset is split into training and testing sets to evaluate the model's performance.

- [Breast Cancer Dataset from Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

## Installation

To run this project, you need to have Python installed on your machine. You can install the required dependencies using `pip`.

```
pip install numpy pandas scikit-learn
```

Requirements
Python 3.x
NumPy
Pandas
Scikit-learn

## Usage

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/Breast-Cancer-Classification-using-Logistic-Regression.git
```

2. Navigate to the project directory:
   cd Breast-Cancer-Classification-using-Logistic-Regression

3. Open and run the Jupyter Notebook:
   jupyter notebook Breast_Cancer_Classification.ipynb

## Model

The model used in this project is a Logistic Regression classifier. The data is preprocessed by splitting it into training and testing sets. Key steps include:

### Data Preprocessing

- Train-Test Split: Splitting the dataset into training and testing sets for model evaluation.

### Model Training

- Logistic Regression: A linear classification model is trained on the processed data to classify breast cancer cases as benign or malignant.

### Evaluation

The model is evaluated using the following metric:

- Accuracy Score: Measures the percentage of correct predictions made by the model. A higher accuracy score indicates better performance in classifying benign and malignant cases.
