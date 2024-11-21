import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder

# create aggregate features
def aggregate_features(df):
    """
    Aggregate features by customer.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to be aggregated.

    Returns
    -------
    df : pandas DataFrame
        The dataframe with the aggregate features added.

    """

    Group_all = df.groupby(['CustomerId'])
    df['TotalTransactionAmount'] = Group_all['Amount'].transform('sum')
    df['AverageTransactionCount'] = Group_all['Amount'].transform('count')
    df['AverageTransactionAmount'] = Group_all['Amount'].transform('mean')
    df['StdTransactionAmount'] = Group_all['Amount'].transform('std')
    df['MinTransactionAmount'] = Group_all['Amount'].transform('min')
    df['MaxTransactionAmount'] = Group_all['Amount'].transform('max')
    return df

def extract_features(df):
    # Ensure the 'TransactionStartTime' column is converted to datetime
    """
    Extracts date and time features from the 'TransactionStartTime' column in the dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe containing the 'TransactionStartTime' column.

    Returns
    -------
    df : pandas DataFrame
        The dataframe with added columns for 'TransactionHour', 'TransactionDay',
        'TransactionMonth', and 'TransactionYear', indicating the respective time features.
        Also prints a warning if any 'TransactionStartTime' entries couldn't be converted.
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    
    # Check if any rows failed to convert
    if df['TransactionStartTime'].isna().any():
        print("Warning: Some entries in 'TransactionStartTime' could not be converted to datetime.")
    
    # Extract features from 'TransactionStartTime'
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year

    return df

# Encode categorical features using one-hot encoding
def encode_categorical_features(df, categorical_cols):
    # Identify categorical columns
    """
    Encode categorical features in a given dataframe using one-hot encoding.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe containing the categorical features to be encoded.

    Returns
    -------
    df_encoded : pandas DataFrame
        The dataframe with the encoded categorical features.

    """
  # Encoding categorical data
def encode_categorical_data(df):
    """
    Encode categorical features in a given dataframe using a LabelEncoder.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe containing the categorical features to be encoded.

    Returns
    -------
    df_encoded : pandas DataFrame
        The dataframe with the encoded categorical features.
    """
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])
    return df

# handle missing values using imputation
def handle_missing_values(df):
    """
    Handle missing values in a given dataframe by imputing the mean of
    corresponding columns.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe containing the missing values to be imputed.

    Returns
    -------
    df_imputed : pandas DataFrame
        The dataframe with the imputed missing values.

    """
    df.fillna(df.mean(), inplace=True)
    return df

# normalize numerical features
def normalize_numerical_features(df):
    """
    Normalize numerical features in a dataframe to have a mean of 0 and standard deviation of 1.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe containing numerical features to be normalized.

    Returns
    -------
    df : pandas DataFrame
        The dataframe with normalized numerical features.
    """
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df
