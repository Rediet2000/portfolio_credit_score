import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from analysis_script import load_data

# create aggregate features
def aggregate_features(df):
    Group_all = df.groupby(['CustomerId'])
    df['TotalTransactionAmount'] = Group_all['Amount'].transform('sum')
    df['AverageTransactionCount'] = Group_all['Amount'].transform('count')
    df['AverageTransactionAmount'] = Group_all['Amount'].transform('mean')
    df['StdTransactionAmount'] = Group_all['Amount'].transform('std')
    df['MinTransactionAmount'] = Group_all['Amount'].transform('min')
    df['MaxTransactionAmount'] = Group_all['Amount'].transform('max')
    return df
aggr = aggregate_features(load_data('data.csv'))

def extract_features(df):
    # Ensure the 'TransactionStartTime' column is converted to datetime
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
extracted = extract_features(load_data('data.csv'))

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
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])
    return df
encoded = encode_categorical_data(load_data('data.csv'))

# Print the first few rows of the encoded DataFrame
print(encoded.head())

# handle missing values using imputation
# def handle_missing_values(df):
#     """
#     Handle missing values in a given dataframe by imputing the mean of
#     corresponding columns.

#     Parameters
#     ----------
#     df : pandas DataFrame
#         The dataframe containing the missing values to be imputed.

#     Returns
#     -------
#     df_imputed : pandas DataFrame
#         The dataframe with the imputed missing values.

#     """
#     df.fillna(df.mean(), inplace=True)
#     return df

# imputed = handle_missing_values(load_data('data.csv'))

# normalize numerical features
def normalize_numerical_features(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df
normalized = normalize_numerical_features(load_data('data.csv'))
print(normalized.head())