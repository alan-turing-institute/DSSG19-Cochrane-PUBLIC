import os
import sys
import argparse

import pandas as pd
from tqdm import tqdm

from src.analyze_results.visualization import plot_precision_recall_curve_best

sys.path.append('../utils/')
sys.path.append('../pipeline/')

from load_config import load_psql_env, load_local_paths, load_config
from SQLConn import SQLConn
from visualization import *


def create_viz_production(results_table_name='final_model_eval'):
    """
    Creates visualizations for models that are trained on the full training data set,
    and are used in production.

    Parameters
    ----------

    results_table_name : str
        The name of a SQL table which contains results (from the test set) about the final
        models.
    """

    # set up required variables
    local_paths_env = load_local_paths('../pipeline/local_paths.yaml')
    env = load_psql_env(local_paths_env['pgpass_path'])
    ignition = load_config(local_paths_env['ignition_path'] + '_1_baseline_ignition.yaml')

    # open sql connection
    connection = SQLConn(env)
    connection.open()

    # pull data from table
    query = f"select * from results.{results_table_name};"
    results_df = pd.read_sql_query(query, connection.conn)
    results_df['label'] = results_df['review_group']

    # create directory for visualizations
    vis_dir = f'{local_paths_env["store_visualizations"]}/production_citations'

    # create folders to store visualizations
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # precision recall plots
    for rg in tqdm(results_df['review_group'].unique()):
        plot = plot_precision_recall_curve_best(results_df, rg, plot_baseline=False)
        plt.savefig(f'{vis_dir}/pr_curve-{rg}.png')
        plt.close()

    # stacked bar workload
    plot = workload_relative_stackedbar(results_df)
    plt.savefig(f'{vis_dir}/workload_relative.png')
    plt.close()

    plot_average_workload_reduction(results_df)
    plt.savefig(f'{vis_dir}/workload_average.png')
    plt.close()


if __name__ == "__main__":
    create_viz_production()
