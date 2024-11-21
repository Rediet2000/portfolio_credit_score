import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# #Add dataset
def  load_data(df):
    """
    Load a dataset from a csv file
    
    Parameters
    ----------
    df : str
        The path to the csv file to be loaded
    Returns
    -------
    dataset : pandas DataFrame
        The loaded dataset
    """
    dataset= pd.read_csv(df)
    return dataset


def overview_dataset(df):
    """
    Perform basic analysis of a dataframe
    
    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to be analyzed
    """
    
    print(df.info())
    print(df.describe())
    print(df.columns)
    print(df.dtypes)
    print(df.isnull().sum())
    print(df.isna().sum())

# Distribution of numerical variables
def plot_numerical_distributions(df):
    """Plot distributions of numerical features."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    n_cols = 2
    n_rows = (len(numerical_cols) + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='blue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Distribution of categorical variables
def plot_categorical_distributions(df):
    """Plot distributions of categorical features."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    n_cols = 2
    n_rows = (len(categorical_cols) + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        sns.countplot(df[col], ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
# Corelation analysis
def corelation_analysis(df):
    """
    Perform corelation analysis of a dataframe
    
    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to be analyzed
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
# Indetify missing values
def missing_values(df):
    """
    Identify missing values in a dataframe
    
    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to be analyzed
    """
    print(df.isnull().sum())
    print(df.isna().sum())
print(missing_values(load_data('data.csv')))

def outlierDetection(df):
    """
    Perform outlier detection on a dataframe
    use box plot to identify outliers
    
    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to be analyzed
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(df[col])
        plt.title(f'Box Plot for {col}')
        plt.show()
outlierDetection(load_data('data.csv'))


 
