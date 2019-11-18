import argparse
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../models/')
sys.path.append('../utils/')
sys.path.append('../analyze_results/')
sys.path.append('../store/')
sys.path.append('../features/')

from load_config import load_local_paths, load_config
from persist import save, load
from etl_features import create_features_production
from text_processing import *


def score_papers(papers, prod_config, models_path):
    """
    Scores papers based on best models for each review group.

    Parameters
    =========-
    papers : pd.DataFrame
        Papers to be scored. One paper is one row.
        Columns are features needed for the classifiers.
    prod_config : dict
        Config file for production pipeline. Includes recall values for each group and
        features to pull into training data.
    models_path : str
        Path to pickled model objects.

    Returns
    =======
    papers_scored : pd.DataFrame
        Scored papers. Contains paper recordid, title, abstract, and 54 columns
        corresponding to the model score for that review group.
    """

    papers_scored = papers.copy()

    for label in tqdm(prod_config['review_groups_recall'].keys(), desc='Score review groups'):
        label_ = label.lower()
        classifier = load(location=models_path, filename=f"prod_models_{label}")
        papers_scored[label_] = classifier.predict(papers)[label_].to_list()

    return papers_scored


def apply_thresholds(papers_scored, upper_thresholds, lower_thresholds):
    """
    Using precision-based upper thresholds and recall-based lower
    thresholds computed during model selection, apply thresholds to
    scored papers and append columns of recommendations.

    Parameters
    ===========
    papers_scored : pd.DataFrame
        Scored papers. Contains paper recordid, title, abstract, and 54 columns
        corresponding to the model score for that review group.
    upper_thresholds : dict
        Dictionary where key=review group and value=upper threshold value.
    lower_thresholds : dict
        Dictionary where key=review group and value=lower threshold value.

    Returns
    =======
    papers_scored_recs : pd.DataFrame
        Same as papers_scored but with 54 more columns with recommended status
        for each paper for each review group ['keep','consider','discard'].
    """

    ### make a copy of the dataframe ###
    papers_scored_recs = papers_scored.copy()

    ### loop through all review groups ###
    for review_group in upper_thresholds:

        ### prep to update dataframe ###
        status_col = f'{review_group}_status'
        status = []

        ### grab thresholds ###
        upper_threshold = upper_thresholds[review_group]
        lower_threshold = lower_thresholds[review_group]

        ### calculate status for paper scores for that review group ###
        for paper_score in papers_scored_recs[review_group]:

            if paper_score >= upper_threshold:

                status.append('keep')

            elif paper_score < upper_threshold and paper_score >= lower_threshold:

                status.append('consider')

            elif paper_score < lower_threshold:

                status.append('discard')

        ### append new column to dataframe with status ###
        papers_scored_recs[status_col] = status

    return papers_scored_recs


def run_scoring_pipeline(papers, features):

    """
    Run pipeline to score each paper. First computes features
    needed to score papers and then scores the papers and generates
    recommended actions based on thresholds produced during model
    selection.

    Parameters
    ==========
    papers : pd.DataFrame
        Papers to be scored. One paper is one row.
        Columns are features needed for the classifiers.
    features : dict
        Dictionary of feature names and functions to create those features.

    Returns
    =======
    None
    """

    # Load local paths file
    local_paths = load_local_paths('local_paths.yaml')

    # Load product config file
    prod_config = load_config('../prod/prod_config.yaml')

    # Compute features
    papers_features = create_features_production(papers=papers)

    # Score papers
    papers_scored = score_papers(papers=papers_features, prod_config=prod_config,
                                 models_path=local_paths['store_production_models'])

    # Apply thresholds
    upper_thresholds = load(local_paths['store_production_models'],'upper_thresholds')
    lower_thresholds = load(local_paths['store_production_models'],'lower_thresholds')
    papers_scored_recs = apply_thresholds(papers_scored=papers_scored,
                                          upper_thresholds=upper_thresholds,
                                          lower_thresholds=lower_thresholds)

    # Save scored papers
    save(object=papers_scored_recs, location=local_paths['store_scored_papers'],
         filename='scored_papers')

    print("Scoring pipeline complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Score new papers.')

    parser.add_argument('--path', type=str, help='Path to directory with new papers to score.')
    parser.add_argument('--file_name', type=str, help='File name of compressed csv with new papers.')

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.path is not None and FLAGS.file_name is not None:
        new_papers = load(FLAGS.path, FLAGS.file_name)

    else:
        print('Could not parse args. Scoring held out test data by default.')
        local_paths = load_local_paths('local_paths.yaml')
        new_papers = load(location=local_paths['store_test_for_scoring'], filename='test_papers_x')

    # run production pipeline
    features = {"tokens_baseline":tokenize_title_abstract_baseline,
                "tokens_no_stopwords":tokenize_title_abstract,
                #"average_embeddings": compute_embeddings,
                "stemmed_tokens_nostop": stemming}
    run_scoring_pipeline(papers=new_papers, features=features)
