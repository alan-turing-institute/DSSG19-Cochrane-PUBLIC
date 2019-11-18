import os
import sys
import pandas as pd

from load_config import load_local_paths
from src.analyze_results.results import pull_results, get_best_hyperparam_algorithm, get_best_hyperparam_all
from src.analyze_results.visualization import plot_precision_recall_curve_hyperparams, \
    plot_optimal_precision_recall_curve, plot_distribution_precision, plot_precision_recall_curve_best

sys.path.append('../utils/')
sys.path.append('../pipeline/')

from tqdm import tqdm


def main(ignition_ids=['1', '2', '4', '5', '7', '10', '15', '16', '17', '18', '19', '20', '21']):
    """
    Script for running a quick visualization of results for models
    trained during crossvalidation and storing these visualizations.

    Parameters
    ==========
    ignition_ids : list
        List of the ignition id for which results are stored, and which
        should be taken into account for visualization. In the list, these
        ignition ids should be stored as strings.
    """

    # load env file containing location to store visualizations
    local_paths_env = load_local_paths('../pipeline/local_paths.yaml')

    best_all_ignitions = []

    for id in tqdm(ignition_ids, desc='Ignition id'):

        vis_dir = f'{local_paths_env["store_visualizations"]}/{id}'

        # create folders to store visualizations
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        # pull results
        results_table = pull_results(ignition_ids=[id], table_name='results.evaluate_rg')

        # calculate best results for this model
        results_best_hyperparam = get_best_hyperparam_algorithm(results_table)
        results_best_hyperparam.reset_index().to_csv(f'{local_paths_env["store_visualizations"]}/results_{id}.csv')
        best_all_ignitions.append(results_best_hyperparam)

        # plots for each review group separately
        for rgroup in results_table['label'].unique():
            # precision-recall curves for all hyperparameters
            plt = plot_precision_recall_curve_hyperparams(results_table, rgroup)
            plt.savefig(f'{vis_dir}/pr_curve_allhyperparam-{rgroup}.png')
            plt.close()

            # precision-recall curves for best hyperparameters
            plt = plot_optimal_precision_recall_curve(results_best_hyperparam, rgroup)
            plt.savefig(f'{vis_dir}/pr_curve_besthyperparam-{rgroup}.png')
            plt.close()

        # plot distribution of precisions at specified recalls
        for recall in [0.9, 0.95, 0.97, 0.99]:
            plot = plot_distribution_precision(results_best_hyperparam, recall)
            plt.savefig(f'{vis_dir}/precision_distribution-recall_{recall}.png')
            plt.close()

    # concatenate best results from each model into one dataframe
    best_all_ignitions = pd.concat(best_all_ignitions)

    # calculate best results across models
    best_overall = get_best_hyperparam_all(best_all_ignitions)
    best_overall.reset_index().drop(columns=['index']).to_csv(
        f'{local_paths_env["store_visualizations"]}/results_overall.csv')

    # create directory for best visualizations
    vis_dir = f'{local_paths_env["store_visualizations"]}/overall'

    # create folders to store visualizations
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    for group in best_overall['label'].unique():
        # precision_recall curves for the best models for each ignition id
        # (so the best hyperparameter combination is chosen for each model)
        plot = plot_precision_recall_curve_best(best_all_ignitions, group)
        plt.savefig(f'{vis_dir}/pr_curve-{group}.png')
        plt.close()

        # precision-recall curve for the top-5 models
        n = 5
        plot = plot_precision_recall_curve_best(best_all_ignitions, group, best_n=n)
        plt.savefig(f'{vis_dir}/pr_curve_best{n}-{group}.png')
        plt.close()


if __name__ == '__main__':
    main()
