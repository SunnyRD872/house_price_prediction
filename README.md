
# House Price Prediction using Machine Learning



## Overview

This project aims to predict house prices using machine learning techniques. It involves data cleaning, outlier removal, feature selection, model training, and deployment using Streamlit.
## Dataset

Dataset is downloaded from here: https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data
## Data Preprocessing

1.Loading the Dataset: 

-The dataset is loaded into a Pandas DataFrame.

2.Cleaning the Data:

-Removing outliers (e.g., bathrooms greater than bedrooms, unusually low-priced multi-BHK houses).

-Dropping unwanted columns.

3.Data Visualization:

-Scatter plots and histograms are used to understand the data distribution.

4.Feature Encoding:

-The Location column is converted into numerical values using One-Hot Encoding.

## Key Features for Price Prediction 

   1.Location (one-hot encoded)

   2.BHK (Number of Bedrooms, Halls, and Kitchens)

   3.Number of Bathrooms

   4.Total Square Feet


## Model Training

The dataset is split into training and testing sets using train_test_split. The following models are tested:

-Linear Regression

-Lasso Regression

-Decision Tree Regressor

# Hyperparameter Tuning

Hyperparameter tuning is performed using GridSearchCV to optimize model performance. The process involves:

-Testing different models, including Linear Regression, Lasso Regression, and Decision Tree Regressor.

-Defining a set of hyperparameters for each model.

-Applying cross-validation to ensure better generalization.

-Evaluating models based on performance metrics such as accuracy and mean squared error.

-Selecting the best model with the highest performance score.


## Model Saving & Deployment

Saving the Model:

-The final trained model is saved using pickle.

Deployment using Streamlit:

-A simple Streamlit app is created to allow users to   input house features and get price predictions.

## Technologies Used

-Python

-Pandas, NumPy (Data Processing)

-Matplotlib, Seaborn (Visualization)

-Scikit-Learn (Machine Learning)

-Streamlit (Web App Deployment)
## Conclusion

This project successfully predicts house prices using machine learning models. Through hyperparameter tuning and feature selection, model performance was improved, and after tuning, Linear Regression emerged as the best model with an accuracy of 0.84522. The trained model was then deployed using Streamlit for easy accessibility, allowing users to input relevant features and obtain price predictions efficiently


## Special Thanks

This project was inspired by the tutorial from the codebasics Channel on YouTube.
Channel:https://www.youtube.com/@codebasics
