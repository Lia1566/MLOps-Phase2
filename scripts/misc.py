"""
A list of miscelaneous functions
"""
from IPython.display import HTML


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

    
        