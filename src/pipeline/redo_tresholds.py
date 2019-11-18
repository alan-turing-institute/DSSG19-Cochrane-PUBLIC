import pandas as pd
import sys
from multiprocessing import Pool
from functools import partial

sys.path.append('../')
sys.path.append('../models/')
sys.path.append('../utils/')
sys.path.append('../analyze_results/')
sys.path.append('../store/')

from load_config import load_local_paths, load_config, load_psql_env
from create_data_sample import sample
from select_classifier import *
from persist import save, load
from SQLConn import SQLConn
from pipeline_scoring import *
from evaluate import *
from visualization import *


def choose_models_with_recall(models_df, group_min_recalls, labels_col='label',
                              recall_col='recall', algorithm_col='algorithm',
                              hyperparameters_col='hyperparameters'):
    """
    Selects algorithm and hyperparameters of best model for a specific
    recall value for each review group according to user input.

    Parameters
    ==========
    models_df : pd.DataFrame
        Dataframe where each row contains information about the best hyperparameters
        for each group and each value of recall evaluated in the ML pipeline.
    group_min_recalls : dict
        Dictionary where keys=review group labels and values=minimum value of recall.
    labels_col : str
        Name of column corresponding to review group labels in models_df.
    recall_col : str
        Name of column corresponding to recall value in models_df.
    algorithm_col : str
        Name of column corresponding to algorithm in models_df.
    hyperparameters_col : str
        Name of column corresponding to hyperparameters in models_df.

    Returns
    =======
    best_models : dict
        Dictionary where key=review group and value=dictionary with best algorithm
        and hyperparameters for that group and specified minimum recall value.
    """

    best_models = {}

    # grab best model hyperparameters for specified value of recall for each label
    for label in group_min_recalls:

        # initialize new dict to store hyperparameters
        best_model = {}

        best_model['algorithm'] = models_df[(models_df[labels_col]==label) & (models_df[recall_col]==group_min_recalls[label])][algorithm_col].values[0]
        best_model['hyperparameters'] = models_df[(models_df[labels_col]==label) & (models_df[recall_col]==group_min_recalls[label])][hyperparameters_col].values[0]

        best_models[label] = best_model

    return best_models


def train_best_models_mp(X_train, y_train, best_models, prod_config, local_paths, cores):
    """
    Multiprocessing pipeline for training best models

    Parameters
    ==========
    X_train : DataFrame
        Training features
    y_train : DataFrame
        Training labels
    best_models : dict
        Dictionary where key=review group and value=dictionary with best algorithm
        and hyperparameters for that group and specified minimum recall value.
    prod_config : dict
        Config file for production pipeline. Includes recall values for each group and
        features to pull into training data.
    local_paths : dict
        Local directory paths, used to store models for production.
    keys: list of key values to subset best models by
    cores: number of cores to use

    Returns
    =======
    None
    """
    #list of lists of keys chunked to number of cores
    key_chunks = [list(best_models.keys())[i::cores] for i in range(cores)]
    #create pool object
    pool = Pool(processes=cores)
    #turn train_best_models into a partial function with keys being variable
    train_best_models_partial = partial(train_best_models, X_train=X_train,
                                        y_train=y_train, best_models=best_models,
                                        prod_config=prod_config, local_paths=local_paths)
    #map function to each core
    pool.map(train_best_models_partial, key_chunks)


def train_best_models(keys, X_train, y_train, best_models, prod_config, local_paths):
    """
    Train and store model objects for each review group.
    Parameters
    ==========
    X_train : DataFrame
        Training features
    y_train : DataFrame
        Training labels
    best_models : dict
        Dictionary where key=review group and value=dictionary with best algorithm
        and hyperparameters for that group and specified minimum recall value.
    prod_config : dict
        Config file for production pipeline. Includes recall values for each group and
        features to pull into training data.
    local_paths : dict
        Local directory paths, used to store models for production.
    keys: list of key values to subset best models by

    Returns
    =======
    None
    """
    # subset best models
    best_models = dict((k, best_models[k]) for k in (keys))

    # Loop through review groups and train model on all data
    #for review_group, params in tqdm(best_models.items(), desc='Training Review Group Production Models'):
    for review_group, params in best_models.items():

        print(f'training {review_group}')

        classifier = select_classifier(classifier_name=params['algorithm'],
                                       fold_hash='prod',
                                       target=None,
                                       classes=list(prod_config['classes']),
                                       model_parts={},
                                       hyperparameters=eval(params['hyperparameters']),
                                       seed=prod_config['seed'],
                                       citations_cols=prod_config['citations_cols'])
        classifier.train(X_train, y_train[[review_group.lower()]])

        # Store models locally
        save(object=classifier, location=local_paths['store_production_models'], filename=f"prod_models_{review_group}")
        print(f'training {review_group} done')


def perform_model_selection(evaluate_best_models=True):
    """
    Run model selection pipeline.
    """
    # Load local paths file
    local_paths = load_local_paths('local_paths.yaml')

    # Load product config file
    prod_config = load_config('../prod/prod_config.yaml', append_static=False)

     # SQL set up
    psql_env = load_psql_env(pgpass_path=local_paths['pgpass_path'])
    connection = SQLConn(psql_env)
    connection.open()

    # Pull data
    X_train, X_test, y_train, y_test = sample(ignition=prod_config, connection=connection,
                                              local_features_path=local_paths['store_features'])

    if evaluate_best_models:

        # Test best models for each review group
        scored_papers_test = load(location=local_paths['store_scored_papers'], filename='scored_papers')

        y_pred_test = scored_papers_test[[col for col in scored_papers_test.columns if col.upper() in prod_config['review_groups_recall'].keys()]]
        y_test = y_test[[col for col in y_test.columns if col.upper() in prod_config['review_groups_recall'].keys()]]

        # calculate thresholds
        upper_thresholds, lower_thresholds, recall_at_thresholds = get_thresholds(y_test, y_pred_test, minimum_recall=0.99)

        # persist thresholds for production
        save(upper_thresholds, local_paths['store_production_models'], 'upper_thresholds')
        save(lower_thresholds, local_paths['store_production_models'], 'lower_thresholds')

        # calculate workload reductions
        keep, consider, discard = get_workload_reduction(y_test, y_pred_test, upper_thresholds, lower_thresholds)

        rg_list = []
        wrkld_reductions = []

        # loop over review groups
        for review_group in tqdm(prod_config['review_groups_recall'].keys(), desc='Review Group'):

            rg = review_group.lower()

            # get thresholds
            thresholds = [upper_thresholds[rg], lower_thresholds[rg]]
            recall_at_threshold = [recall_at_thresholds[rg]['upper'], recall_at_thresholds[rg]['lower']]
            workload_reduction = [keep[rg], consider[rg], discard[rg]]

            rg_list.append(rg)
            wrkld_reductions.append(workload_reduction)

        d = {'review_group':rg_list, 'workload_reduction':wrkld_reductions}
        df = pd.DataFrame.from_dict(d)
        plot_average_workload_reduction(df)



    connection.close()


    print("Model selection pipeline complete.")


if __name__ == "__main__":

    perform_model_selection()
