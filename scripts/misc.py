"""
A list of miscelaneous functions
"""
from IPython.display import HTML
import pandas as pd
import os
import sys

_strong = '<strong>{}</strong>'
_break = '<hr><br>'

def get_relevant_paths(ROOT):
    """
    Based on the ROOT path, then return relevant paths for the flow.
    """
    # Set the paths for relevant data
    DATA_RAW = os.path.join(ROOT, "data", "raw")
    DATA_PROC = os.path.join(ROOT, "data", "processed")
    FIGS = os.path.join(ROOT, "reports", "figures")

    # Create output directories 
    for path in [DATA_PROC, FIGS]:
        os.makedirs(path, exist_ok=True)

    print(f'Project Root: {ROOT}')
    print(f'Raw Data: {DATA_RAW}')
    print(f'Processed Data: {DATA_PROC}')
    print(f'Figures: {FIGS}')

    return DATA_RAW, DATA_PROC, FIGS

# ---
# PREPROCESSING
# ---
def drop_duplicates(df):
    """
    A function that takes a dataframe and drops the duplicates, returns a cleaned dataframe
    with no duplicates.

    dataframe: The dataframe to describe. Expects a pandas-like dataframe.
    """
    display(HTML('<strong>Drop duplicates</strong>'))
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    # Remove duplicates
    df_clean = df.copy().drop_duplicates()
    
    print(f"After removing duplicates: {df_clean.shape}")
    print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")
    print(f"Percentage removed: {((df.shape[0] - df_clean.shape[0]) / df.shape[0] * 100):.2f}%")
    
    # Verify no missing values
    print(f"\nMissing values check:")
    print(df_clean.isnull().sum().sum())
    print("✓ No missing values" if df_clean.isnull().sum().sum() == 0 else "Missing values found")

    return df_clean


def create_binary_target(df):
    """
    A funciton that takes a dataframe and adds a new column with a transformed binary
    target from a text-like to numerical.

    dataframe: The dataframe to describe. Expects a pandas-like dataframe.
    """

    display(HTML('<strong>Create Binary Target Variable</strong>'))
    def _create_binary_target(performance):
        """
        Convert 4-class performance to binary
        - Excellent, Vg  → 1 (High Performance)
        - Good, Average → 0 (Lower Performance)
        """
        if performance in ['Excellent', 'Vg']:
            return 1 # High performance
        else:
            return 0 # Lower performance

    # Create a copy of the curreint input dataframe
    df_clean = df.copy()

    # Original Performance distribution
    print("\nOriginal Performance Distribution:")
    print(df_clean['Performance'].value_counts())
    print("\nPercentages:")
    print(df_clean['Performance'].value_counts(normalize=True).mul(100).round(2))
    
    # Create bianry target varibale
        
    df_clean['Performance_Binary'] = df_clean['Performance'].apply(_create_binary_target)
    
    # Check binary distribution
    print(f"\n{'='*80}")
    print("Binary Performance Distribution:")
    print(f"{'='*80}")
    print(df_clean['Performance_Binary'].value_counts())
    print("\nClass Balance:")
    balance = df_clean['Performance_Binary'].value_counts(normalize=True).mul(100).round(2)
    print(balance)
    
    print(f"\nClass 0 (Lower Performance): {balance[0]}%")
    print(f"Class 1 (High Performance): {balance[1]}%")
    print(f"Balance Ratio: {balance[0]/balance[1]:.2f}:1")

    return df_clean

    
# ---
# EXPLORATORY DATA ANALYSIS
# ---

def describe_dataframe(df):
    """
    A function that imports the dataset and returns a basic EDA.

    dataframe: The dataframe to describe. Expects a pandas-like dataframe.
    """

    _strong = '<strong>{}</strong>'
    _break = '<hr><br>'
    
    # Get basic information of the dataset
    display(HTML('<strong>Basic information<strong>'), HTML('<br>'))
    print(f'Dataset Shape: {df.shape}')
    print(f'Number of Rows: {df.shape[0]}')
    print(f'Number of Columns: {df.shape[1]}')
    print('Data information:')
    display(df.info())
    display(HTML(_break))

    # Get a sample of the dataframe, just 1 row.
    display(HTML('<strong>Get a sample of the dataframe</strong>'), HTML('<br>'))
    _ = df.head(1).T.reset_index().rename(columns={'index':'COLUMNS', 0:'VALUE'})
    display(_, HTML(_break))

    # Basic statiscs
    display(HTML(_strong.format('Basic statistics')))
    display(df.describe(include='all'))
    display(HTML(_break))


def get_missing_values(df):
    """
    A function that takes a dataframe and returns the number of missing values.

    df: The dataframe to describe. Expects a pandas-like dataframe.
    """

    display(HTML('<strong>Missing Values Analysis</strong>'))

    # Get the missing count
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })

    print(missing_df)
    print(f'\nTotal missing values: {missing.sum()}')


def get_duplicates_info(df):
    """
    A function that looks for duplicate information.

    df: The dataframe to describe. Expects a pandas-like dataframe.
    """

    display(HTML('<strong>Duplicate Analysis</strong>'))
    print(f'Number of duplicate rows: {df.duplicated().sum()}')
    print(f'Percentage of duplicate rows: {(df.duplicated().sum() / len(df)) * 100:.2f}%')


def get_unique_values_per_columns(df):
    """
    A function that returns the unique values per column.

    df: The dataframe to describe. Expects a pandas-like dataframe.
    """
    
    display(HTML('<strong>Unique Values per Columns</strong>'))
    for col in df.columns:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Values: {df[col].unique()[:10]}")  # Show first 10


def get_value_counts_per_columns(df):
    """
    A function that returns the value counts for each of the categorical column,
    df: The dataframe to describe. Expects a pandas-like dataframe.
    """

    display(HTML('<strong>Value Counts for Each Column<strong>'))
    for col in df.columns:
        print(f"\n{'='*80}")
        print(f"{col.upper()}")
        print(f"{'='*80}")
        print(df[col].value_counts())
        print()

















    
    
        