import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

extract = extract_features(load_data('data.csv'))
print(extract)

    