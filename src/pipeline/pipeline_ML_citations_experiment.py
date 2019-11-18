# Pipeline
import argparse
import sys

sys.path.append('../utils/')
sys.path.append('../models/')
sys.path.append('../features/')
sys.path.append('../store/')
sys.path.append('../dataproc/')

from load_config import load_config, load_psql_env, load_local_paths
from create_data_sample import sample
from select_classifier import select_classifier
from evaluate import compute_metrics, results_to_db
from persist import create_hash_id, check_persisted, load, save
from ml_pipe_tools import expand_grid
from cross_validation import k_fold
from SQLConn import SQLConn
from tqdm import tqdm


def run_pipeline(ignition_file, persist_all, load_all_fresh):
    """
    An adhoc pipeline created to mirror the standard ML pipeline and work
    with citations data.

    Parameters:
    ===========
    ignition_file: string
        name of the yaml file for which you want to run an experiment

    persist_all: boolean
        T if you want to persist all data for future use

    load_all_fresh: boolean
        T if you want to avoid any persisted data and load new data from scrath

    Returns:
    ========
    None
    """

    model_parts = {}

    ##### 1. LOAD ENVIRONMENT DATA #####

    # load local paths
    local_paths_env = load_local_paths('local_paths.yaml')
    print('Local paths loaded.')

    # load ignition file
    ignition = load_config(local_paths_env['ignition_path'] + ignition_file)
    print('Ignition loaded.')

    # id used for persisting
    hash_id = create_hash_id(str(ignition['id']))
    print('Hash id created.')

    # create hyperparameter combinations (for k-folding)
    hyperparameters = expand_grid(ignition['hyperparameters'])

    # load environment file
    psql_env = load_psql_env(pgpass_path=local_paths_env['pgpass_path'])
    print('PSQL environment file loaded.')

    # Initiate PSQL Connection
    connection = SQLConn(psql_env)
    connection.open()

    ##### 2. LOAD TRAIN AND TEST DATA #####

    if check_persisted(local_paths_env['store_train_data'], f'{hash_id}_x', load_all_fresh):

        print("Found data")

        # data loaded before: load from file
        X_train = load(local_paths_env['store_train_data'], f'{hash_id}_x')
        X_test = load(local_paths_env['store_test_data'], f'{hash_id}_x')
        y_train = load(local_paths_env['store_train_data'], f'{hash_id}_y')
        y_test = load(local_paths_env['store_test_data'], f'{hash_id}_y')

        print('Loaded data from file.')

    else:

        print("Data not found in storage - load from database")

        # data not loaded: pull from database and create features
        X_train, X_test, y_train, y_test = sample(ignition, connection, local_paths_env['store_features_citations_only'])
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        # add fold index column to data
        X_train, y_train = k_fold(X_train, y_train, ignition['k_folds'], ignition['k_folds_seed'])

        # save data to file for future use
        save(X_train, local_paths_env['store_train_data'], f'{hash_id}_x', persist_all)
        save(X_test, local_paths_env['store_test_data'], f'{hash_id}_x', persist_all)
        save(y_train, local_paths_env['store_train_data'], f'{hash_id}_y', persist_all)
        save(y_test, local_paths_env['store_test_data'], f'{hash_id}_y', persist_all)

    print('Data loading completed.')


    ##### 3. K-FOLDING #####

    # loop over folds
    for fold in tqdm(range(ignition['k_folds']), desc='Folds'):

        # get fold id hash (for persisting)
        fold_id = create_hash_id(str(ignition['id']) + str(fold))

        # get fold data
        fold_X_train = X_train[X_train['k'] != fold]
        fold_X_test = X_train[X_train['k'] == fold]
        fold_y_train = y_train[y_train['k'] != fold]
        fold_y_test = y_train[y_train['k'] == fold]

        # store fold features, if any
        fold_features = {}

        ##### 4. LOOP OVER HYPERPARAMETERS: TRAIN CLASSIFIER #####

        for hyperparam in tqdm(hyperparameters, desc='Hyperparameters'):

            # create hyperparam unique id and hyperparam-fold unique id
            hyperparam_id = create_hash_id(str(ignition['id']) + str(hyperparam))
            hyperparam_fold_id = create_hash_id(str(ignition['id']) + str(hyperparam) + str(fold))

            # if not check_val_in_db(connection, ignition['results_table_name'],
            # 'results', 'hash_id', hyperparam_fold_id, len(ignition['recalls'])):

            # create classifier of specified type and with specified target
            classifier = select_classifier(ignition["model_type"], fold_id,
                                           ignition["target"],
                                           ignition["classes"], fold_features,
                                           citations_cols=ignition["citations_cols"],
                                           hyperparameters=hyperparam,
                                           seed=ignition['seed'],
                                           env=local_paths_env)
            #print('Classifier created.')

            # train classifier
            classifier.train(fold_X_train, fold_y_train)

            ##### 5. TEST CLASSIFIER #####

            # generate predictions from classifier
            y_probs = classifier.predict(fold_X_test)

            ##### 6. EVALUATION #####

            for recall in tqdm(ignition['recalls'], desc='Evaluations'):

                # compute evaluation metrics
                all_metrics = compute_metrics(metric_names=ignition['metrics'], y_true=fold_y_test.drop(columns=['k']), y_pred=y_probs, k=recall)

                # store results in database
                unique_id = create_hash_id(str(ignition['id']) + str(hyperparam) + str(fold) + str(recall))

                results_to_db(metrics=all_metrics, table_name=ignition['results_table_name'],
                    ignition_id=ignition['id'], hash_id=hyperparam_fold_id, algorithm=ignition['model_type'],
                    hyperparameters=hyperparam, fold=str(fold), recall=recall, unique_id=unique_id,
                    connection=connection)

    connection.close()
    print(f"Done running pipeline for ignition id: {ignition['id']}!")


def print_flags(FLAGS):
    """
    Prints all of the flags which are specified by the function

    Parameters:
    ==========
    FLAGS: dict
        Flags and values that need to be printed

    Returns:
    =======
    Prints flags
    """

    for key, value in vars(FLAGS).items():
        print(f'{key}: {value}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run an experiment as defined in a yaml file.')

    parser.add_argument('--ignition_file', type=str, default='_1_baseline_ignition.yaml', help='Ignition file')
    parser.add_argument('--persist_all', type=bool, default=True, help='Determines whether files should be persisted or not')
    parser.add_argument('--load_all_fresh', type=bool, default=False, help='Determines whether files should be generated anew (even if stored in disk)')

    FLAGS, unparsed = parser.parse_known_args()

    print_flags(FLAGS)

    run_pipeline(ignition_file=FLAGS.ignition_file, persist_all=FLAGS.persist_all, load_all_fresh=FLAGS.load_all_fresh)
