from tkinter import _test
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from analysis_script import *
from sklearn.model_selection import GridSearchCV, train_test_split

# split data into train and test sets
def split_data(df):
    """
    Split the data into training and test sets, apply preprocessing to both and return.
    
    Parameters
    ----------
    df : DataFrame
        The data to be split into training and test sets.
    
    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of numpy arrays
        The preprocessed training and test sets.
    
    """
    
    X = df.drop('FraudResult', axis=1)
    y = df['FraudResult']

    # Identify categorical columns (you might need to adjust this based on your dataset)
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

    # Preprocessing for numerical columns: Impute missing values and scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), 
    ])

    # Preprocessing for categorical columns: Encode categorical variables
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
        ('encoder', OneHotEncoder(handle_unknown='ignore')) 
    ])

    # Combine both transformers into a single column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, X.select_dtypes(include=['float64', 'int64']).columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply preprocessing to both train and test sets
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test

    # train model using logistic regression
def train_model(X_train, y_train):
    """
    Train a logistic regression model using the given training data.

    Parameters
    ----------
    X_train : pandas DataFrame
        The feature variables to be used in the model.
    y_train : pandas Series
        The target variable to be predicted.

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression
        A trained logistic regression model.
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model
# Hyperparameter tuning
def hyperparameter_tuning(model, X_train, y_train):
    """
    Hyperparameter tuning for a logistic regression model.
    imporve model performance by grid search

    Parameters
    ----------
    """
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

# evaluate model performance
def evaluate_model(model, X_test, y_test):
    
    """
    Evaluate the performance of a logistic regression model.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        The trained logistic regression model.
    X_test : pandas DataFrame
        The feature variables to be used for evaluation.
    y_test : pandas Series
        The target variable to be predicted.

    Returns
    -------
    accuracy : float
        The accuracy score of the model.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1_score = f1_score(y_test, y_pred)
    ROC_AUC = roc_auc_score(y_test, y_pred)
    return accuracy
# predict new data
def predict_new_data(model, X_new):
    """
    Predict new data using a trained logistic regression model.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        The trained logistic regression model.
    X_new : pandas DataFrame
        The feature variables to be used for prediction.

    Returns
    -------
    y_pred : numpy array
        The predicted values from the logistic regression model.
    """
    y_pred = model.predict(X_new)
    return y_pred
