from itertools import product
import pandas as pd


def expand_grid(dict):
    """
    Finds all possible combinations of variables and returns a pd.df

    Parameters
    ==========
    dict : hyperparameters where the keys are hyperparameter name and the
    values at the hyperparameter space we're interested in

    Returns
    =======
    List of dictionaries with hyperparameters
    """

    df = pd.DataFrame(product(*dict.values()), columns=dict.keys())
    return df.to_dict(orient='records')


def check_val_in_db(connection, table, schema, key, value, n=1):
    """
    Check if a row already exists in the results table in the
    PSQL database.

    Note: this functions is intended to be used during the machine
    learning pipeline so that the pipeline does not have to re-train
    a model for which we already successfully computed results.
    **It is not currently being used in pipeline_ML.py due to a bug**.

    Parameters
    ==========
    connection: active SQLConn class connection
    table : str
        Name of table in database.
    schema: str
        Name of schema in db which contains the table
    key: str
        Column which we are searching.
    value: str
        Value in 'key' which we are searching for
    n: int
        Number of entries we want

    Returns
    =======
    boolean : whether row was found in database
    """


    #check to see if table exists
    tables = connection.query(f"""select * from
                            information_schema.tables where table_name='{table}'
                            and table_schema='{schema}';""")

    if len(tables.index) == 0:
        return False

    vals = connection.query(f"""
                            select * from {schema}.{table} where
                            {key}='{value}';
                            """)
                            
    if not len(vals.index) == n:
        return False
    else:
        return True
