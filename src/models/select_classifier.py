import sys
sys.path.append('../models')

from baseline_rg_classifier import BaselineClassifier
from elastic_net import ElasticClassifier
from rf_embeddings import RandomForestEm
from rf_tfidf import RandomForestTFIDF
from elastic_net_embeddings import ElasticClassifierEm
from rf_weighted_embeddings import RandomForestWeightedEmbeddings
from xgboost_classifier import XGBoost
from elasticnet_tfidf_pca import ElasticClassifierTfidfPCA
from lgbm_classifier import LightGBM
from adaboost_classifier import AdaBoost
from elastic_net_embeddings_pca import ElasticClassifierEmPCA


def select_classifier(classifier_name, fold_hash, target="inregister",
classes=[], model_parts={}, citations_cols=None, hyperparameters=None,
seed=2019, env=None, load_fresh=False):

    """
    Takes in parameters from pipeline and returns a classifier object.

    Parameters
    ==========
    classifier_name : str
        Name f the classifier as defined in the yaml
    target : str  
    classes : list
    model_parts : dict
        More static model parts that don't depend on hyperparameter specification.
    citations_cols : list
        Columns that contain citation data. 
    hyperparameters : dict
        One set of hyperparameters.
    seed : int
    env : dict
    load_fresh : bool
        Specifies whether all model parts that could be stored on disk should be regenerated (True) or loaded from disk (False)

    Returns
    =======
    classifier : object
    """

    if classifier_name == "baseline":

        return BaselineClassifier(fold_hash, env=env, load_fresh=load_fresh, alpha=hyperparameters['alpha'])

    elif classifier_name == 'elasticnet':

        return ElasticClassifier(fold_hash, alpha=hyperparameters['alpha'], env=env, load_fresh=load_fresh,
                                    tokens_col="tokens_baseline", l1_ratio=hyperparameters['lambda'])

    elif classifier_name == 'elasticnet_nostop':

        return ElasticClassifier(fold_hash, alpha=hyperparameters['alpha'], env=env, load_fresh=load_fresh,
                                    l1_ratio=hyperparameters['lambda'], tokens_col="tokens_no_stopwords")

    elif classifier_name in ['elasticnet_nostop_simple', 'elasticnet_nostop_simple2', 'elasticnet_nostop_simple_full']:

        return ElasticClassifier(fold_hash, tokens_col="tokens_no_stopwords", env=env, load_fresh=load_fresh,
                                    **hyperparameters)

    elif classifier_name in ['elasticnet_nostop_simple_journals']:

        return ElasticClassifier(fold_hash, tokens_col="tokens_no_stopwords", tokens_col2='so', 
                                    env=env, load_fresh=load_fresh, **hyperparameters)

    elif classifier_name in ['elasticnet_nostop_simple_journals_stemmed']:

        return ElasticClassifier(fold_hash, tokens_col="stemmed_tokens_nostop", tokens_col2='so', 
                                    env=env, load_fresh=load_fresh, **hyperparameters)

    elif classifier_name in ['elasticnet_nostop_simple_stemmed']:

        return ElasticClassifier(fold_hash, tokens_col="stemmed_tokens_nostop", 
                                    env=env, load_fresh=load_fresh, **hyperparameters)

    elif classifier_name in ['elasticnet_cited']:

        return ElasticClassifier(fold_hash, citations_cols=citations_cols, 
                                    env=env, load_fresh=load_fresh, **hyperparameters)

    elif classifier_name in ['elasticnet_tfidf_cited']:

        return ElasticClassifier(fold_hash, tokens_col="tokens_no_stopwords", citations_cols=citations_cols,
                                    env=env, load_fresh=load_fresh, **hyperparameters)

    elif classifier_name in ['elasticnet_tfidf_journal_cited']:

        return ElasticClassifier(fold_hash, tokens_col="tokens_no_stopwords", tokens_col2="so", 
                                    env=env, load_fresh=load_fresh, citations_cols=citations_cols, **hyperparameters)

    elif classifier_name == 'elasticnet_nostop_pca':

        return ElasticClassifierTfidfPCA(fold_hash, seed, tokens_col="tokens_no_stopwords",
                                    env=env, load_fresh=load_fresh, **hyperparameters)

    elif classifier_name == 'elasticnet_embeddings_pca':

        return ElasticClassifierEmPCA(fold_hash, seed, embeddings_col="average_embeddings",
                                    env=env, load_fresh=load_fresh, **hyperparameters)

    elif classifier_name == 'rf_em':

        return RandomForestEm(fold_hash, n_estimators=hyperparameters['n_estimators'],
                            max_depth=hyperparameters['max_depth'],
                            max_features=hyperparameters['max_features'],
                            min_samples_leaf=hyperparameters['min_samples_leaf'],
                            n_jobs=hyperparameters['n_jobs'],
                            seed=seed)

    elif classifier_name == 'rf_tfidf':

        return RandomForestTFIDF(fold_hash, n_estimators=hyperparameters['n_estimators'],
                            max_depth=hyperparameters['max_depth'],
                            max_features=hyperparameters['max_features'],
                            min_samples_leaf=hyperparameters['min_samples_leaf'],
                            n_jobs=hyperparameters['n_jobs'],
                            seed=seed, env=env, load_fresh=load_fresh)

    elif classifier_name == 'elasticnet_embeddings':

        return ElasticClassifierEm(alpha=hyperparameters['alpha'],
                                   l1_ratio=hyperparameters['lambda'],
                                   embeddings_col='average_embeddings',
                                   env=env, load_fresh=load_fresh)

    elif classifier_name == 'xgboost':

        return XGBoost(fold_hash, env=env, load_fresh=load_fresh, **hyperparameters)

    elif classifier_name == 'rf_weighted_embeddings':

        return RandomForestWeightedEmbeddings(fold_hash, env=env, n_estimators=hyperparameters['n_estimators'],
                            max_depth=hyperparameters['max_depth'],
                            max_features=hyperparameters['max_features'],
                            min_samples_leaf=hyperparameters['min_samples_leaf'],
                            seed=seed, n_jobs=hyperparameters['n_jobs'],
                            load_fresh=load_fresh)

    elif classifier_name == 'lgbm':

        return LightGBM(fold_hash, env=env, load_fresh=load_fresh, **hyperparameters)

    elif classifier_name == "adaboost":

        return AdaBoost(fold_hash=fold_hash, seed=seed, env=env, load_fresh=load_fresh, **hyperparameters)

    elif classifier_name == "rf_nostop_simple_journals":

        return RandomForestTFIDF(fold_hash=fold_hash, seed=seed, tokens_col="tokens_no_stopwords", tokens_col2='so',
                    env=env, load_fresh=load_fresh, **hyperparameters)

    else:
        raise ValueError("Name of classifier received is not recognized.")
