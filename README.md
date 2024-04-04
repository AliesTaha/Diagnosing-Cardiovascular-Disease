# Diagnosing Cardiovascular Disease- Trees Ensemble

## Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Setup](#setup)
- [One-hot Encoding](#one-hot-encoding)
- [Model Building](#model-building)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
  - [XGBoost](#xgboost)
- [Results and Discussion](#results-and-discussion)
- [Conclusion](#conclusion)

## Introduction
This project builds and compares three different models to predict the likelihood of cardiovascular disease. These models include a Decision Tree, Random Forest, and XGBoost, implemented using scikit-learn and the XGBoost library. The performance of these models is evaluated based on how different parameters affect their accuracy and predictive capabilities.

## Dataset Overview
The dataset used is the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) from Kaggle. It consists of 11 features that can be utilized to predict a possible heart disease occurrence. These features include Age, Sex, ChestPainType, RestingBP, Cholesterol, and more, containing both numerical and categorical variables.

## Setup
Required libraries:

- NumPy
- pandas
- scikit-learn
- XGBoost
- Matplotlib

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
```

## One-hot Encoding
One-hot encoding is applied using pandas to convert categorical variables into a format suitable for the machine learning models. The variables encoded include Sex, ChestPainType, RestingECG, ExerciseAngina, and ST_Slope.

## Model Building
### Decision Tree
A Decision Tree model is implemented with varying min_samples_split and max_depth to observe their impact on performance.
![image](https://github.com/AliesTaha/Diagnosing-Cardiovascular-Disease/assets/103478551/c42988d3-aa27-45a7-8776-dd562768e1b2)
Note how increasing the the number of min_samples_split reduces overfitting.
Increasing min_samples_split from 10 to 30, and from 30 to 50, even though it does not improve the validation accuracy, it brings the training accuracy closer to it, showing a reduction in overfitting.

### Random Forest
A Random Forest model explores the effects of n_estimators, min_samples_split, and max_depth on accuracy.
![image](https://github.com/AliesTaha/Diagnosing-Cardiovascular-Disease/assets/103478551/7fde8368-e089-493d-8fc0-c99bae8929e1)
Notice that, even though the validation accuraty reaches is the same both at min_samples_split = 2 and min_samples_split = 10, in the latter the difference in training and validation set reduces, showing less overfitting.

### XGBoost
The XGBoost model is fitted, experimenting with n_estimators, learning_rate, and early stopping rounds to optimize performance.
Even though we initialized the model to allow up to 500 estimators, the algorithm only fit 26 estimators (over 26 rounds of training).

## Results and Discussion
The accuracy obtained from each model is presented, along with a discussion on overfitting/underfitting observed with varying parameters. Insights into which model performed best for this specific dataset are provided.

## Conclusion
The project concludes with a summary of findings, highlighting the model that best predicts cardiovascular disease occurrence and mentioning potential improvements or further investigations to enhance model performance. We see that XG Boost had the best accuracy. 

## Credit
This project was completed as part of Stanford's Advanced Learning Algorithms [course](https://github.com/AliesTaha/Stanford-ML) 
