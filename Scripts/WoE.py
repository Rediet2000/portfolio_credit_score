import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scorecardpy as sc
from analysis_script import load_data

def rfms_score(df):
    """
    Calculate RFMS score for each customer.
    RFMS: Recency, Frequency, Monetary, Size
    """
    # Ensure 'TransactionStartTime' is in datetime format
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    current_date = df['TransactionStartTime'].max()

    # Aggregate customer metrics
    customer_metrics = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (current_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': ['sum', 'mean'],  # Monetary and Size
    })

    customer_metrics.columns = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg']

    # Normalize the metrics
    for col in customer_metrics.columns:
        customer_metrics[f'{col}_Normalized'] = (customer_metrics[col] - customer_metrics[col].min()) / (customer_metrics[col].max() - customer_metrics[col].min())

    # Calculate RFMS score (weights can be adjusted as needed)
    customer_metrics['RFMS_Score'] = (
        0.25 * (1 - customer_metrics['Recency_Normalized']) +  # Inverse of Recency
        0.25 * customer_metrics['Frequency_Normalized'] +
        0.25 * customer_metrics['MonetaryTotal_Normalized'] +
        0.25 * customer_metrics['MonetaryAvg_Normalized']
    )

    return customer_metrics

# Plot histogram of RFMS metrics
def plot_histogram(customer_metrics):
    """
    Plot histograms of the RFMS metrics and RFMS score.
    """
    plt.figure(figsize=(15, 12))

    metrics = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg', 'RFMS_Score']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        sns.histplot(customer_metrics[metric], kde=True)
        plt.title(f'{metric} Distribution')
        plt.xlabel(metric)
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def divide_good_bad(df, rfms_scores, threshold=0.4):
   
    """
    Categorize customers into 'good' or 'bad' based on RFMS score and merge relevant metrics.

    Parameters
    ----------
    df : pandas DataFrame
        The original dataframe containing customer transaction data.
    rfms_scores : pandas DataFrame
        DataFrame containing RFMS scores and metrics for each customer.
    threshold : float, optional
        The threshold above which customers are labeled as 'good'. Default is 0.4.

    Returns
    -------
    pandas DataFrame
        The dataframe with additional columns for RFMS score, label, and customer metrics.
    """
    customer_labels = rfms_scores['RFMS_Score'].reset_index()
    customer_labels['label'] = np.where(customer_labels['RFMS_Score'] > threshold, 'good', 'bad')

    # Merge with the original dataframe
    df = df.merge(customer_labels[['CustomerId', 'RFMS_Score', 'label']], on='CustomerId', how='left')
    
    customer_metrics = rfms_scores[['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg']].reset_index()
    df = df.merge(customer_metrics, on='CustomerId', how='left')
    
    return df

# Perform WoE binning
def woe_binning(df, target_col, features):
  
    """
    Perform WoE binning on a given dataframe and list of features.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to be binned.
    target_col : str
        The name of the target variable.
    features : list of str
        The list of feature names to be binned.

    Returns
    -------
    tuple
        A tuple containing the binned dataframe and a dictionary of IV values for each feature.
    """

    bins = sc.woebin(df, y=target_col, x=features)
    woe_df = sc.woebin_ply(df, bins)
    
    iv_values = {}
    for feature in features:
        iv_values[feature] = bins[feature]['total_iv'].values[0]
        print(f"IV for {feature}: {iv_values[feature]}")

    return woe_df, iv_values

# Plot WoE binning results for a feature
def plot_woe_binning(bins, feature):
    """
    Plot WoE binning results for a feature.

    Parameters
    ----------
    bins : dict
        The output from sc.woebin
    feature : str
        The feature name to plot

    Returns
    -------
    None
    """
    # Check if the feature exists in the bins and plot the WoE binning.
    if feature in bins:
        plt.figure(figsize=(10, 6))
        sc.woebin_plot(bins[feature])
        plt.title(f'WoE Binning Plot for {feature}')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Feature '{feature}' not found in WoE bins. Available features are:")
        print(bins.keys())

    