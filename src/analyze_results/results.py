import sys
import pandas as pd

sys.path.append('../utils/')
from load_config import load_psql_env, load_local_paths, load_config
from SQLConn import SQLConn

def pull_results(ignition_ids, table_name='results.evaluate_rg',
    metric_col='metrics', metrics=['precision_at_recall'],
    other_cols=['ignition_id','hash_id, algorithm', 'hyperparameters', 'fold', 'recall']):
    """
    Pull results from PSQL table into long dataframe.

    Parameters
    ==========
    ignition_ids : list
        List of ignition_ids to pull into table.
    table_name : str
        Name of PSQL table with results.
    metric_col : str
        Name of column where metrics exist.
    metrics : list
        Metrics to be included in table. Will be parsed from jsonb.
    other_cols : list
        List of other columns to included in table as is.
    labels : list
        Labels to be included in table.

    Returns
    =======
    results_df : pd.DataFrame
        Long dataframe with results from specified ignition files, metrics, and labels.
    """

    local_paths_env = load_local_paths('../pipeline/local_paths.yaml')
    env = load_psql_env(local_paths_env['pgpass_path'])
    ignition = load_config(local_paths_env['ignition_path'] + '_1_baseline_ignition.yaml')

    # establish SQL Connection
    connection = SQLConn(env)
    connection.open()

    ### Set up ###
    results = {}
    ignition_ids_sql = "('" + "','".join(ignition_ids) + "')"
    other_cols_sql = ",".join(other_cols)

    ## Make one query for each label and store resulting df in a dict ###
    i = 0
    for label in ignition['classes']:
        metrics_sql = f"'{label}' as label"
        for metric in metrics:
            metrics_sql += f",{metric_col} -> '{metric}' -> '{label.lower()}' as {metric}"

        qy = f"""
        select {other_cols_sql},
        {metrics_sql}
        from
        {table_name}
        where ignition_id in {ignition_ids_sql};
        """

        results[label] = pd.read_sql_query(qy, connection.conn)

    ## Concatenate all dfs into one long df ###
    results_df = pd.concat(results.values(), ignore_index=True)

    connection.close()

    return results_df


def get_best_hyperparam_algorithm(results_df):

    """
    Given a results table, outputs the best hyperparameter value at each recall score.

    Parameters
    ==========

    results_df : DataFrame
        DataFrame containing pipeline_ML results
    """

    # cannot be a dict
    results_df['hyperparameters'] = results_df['hyperparameters'].astype(str)

    # average hyperparameter scores across folds
    average_folds = results_df.groupby(['algorithm', 'hyperparameters', 'label',
                        'recall']).mean().reset_index()

    # gets highest precision for each recall score (selecting the best hyperparam)
    # in case of ties, the first one in the dataframe is selected
    best_hyperparam = average_folds.loc[average_folds.groupby(['algorithm',
        'label', 'recall'])['precision_at_recall'].idxmax()]

    return best_hyperparam


def get_avg_ignition(results_df, ignition_id='1'):
    """
    Given a results table, outputs the average per algorithm
    defaults to "baseline"

    Parameters
    ==========

    results_df : DataFrame
        DataFrame containing pipeline_ML results.
    ignition_id: str
        Ignition id for which results should be calculated.

    """
    # cannot be a dict
    results_df['hyperparameters'] = results_df['hyperparameters'].astype(str)

    # subset to baseline
    baseline = results_df[results_df['ignition_id']==ignition_id]

    # average hyperparameter scores across folds
    average_folds = baseline.groupby(['algorithm', 'hyperparameters', 'label',
                        'recall']).mean().reset_index()

    best_baseline = average_folds.loc[average_folds.groupby(['label', 'recall'])['precision_at_recall'].idxmax()]

    return best_baseline


def get_best_hyperparam_all(results_df):
    """
    Given a results table with the scores of multiple models, outputs the best model
    + hyperparameter at each recall score.

    Parameters
    ==========

    results_df : DataFrame
        DataFrame containing pipeline_ML results.
    """

    best_overall = results_df.loc[results_df.groupby(['label', 'recall'])['precision_at_recall'].idxmax()]

    return best_overall


def get_best_algorithm_hyperparameter_onestep(results_df):

    """
    Combines get_best_algorithm_hyperparameter() and
    get_best_hyperparam_all() into one function.

    Parameters
    ==========

    results_df : DataFrame
        DataFrame containing pipeline_ML results.

    """
    # Set up
    average_group_cols = ['algorithm', 'hyperparameters', 'label', 'recall', 'ignition_id']
    best_hyperparam_group_cols = ['label', 'recall']
    metric = 'precision_at_recall'

    # Do
    results_df['hyperparameters'] = results_df['hyperparameters'].astype(str)
    average_folds = results_df.groupby(average_group_cols).mean().reset_index()
    best_hyperparam = average_folds.loc[average_folds.groupby(best_hyperparam_group_cols)[metric].idxmax()]

    return best_hyperparam
