import sys

import pandas as pd

sys.path.append('../utils/')

from text_processing import *
from embeddings import *
from tqdm import tqdm


def create_features_etl(conn, file_location,
                        features={"tokens_baseline":tokenize_title_abstract_baseline,
                                  "tokens_no_stopwords":tokenize_title_abstract,
                                  "average_embeddings": compute_embeddings,
                                  "stemmed_tokens_nostop": stemming},
                        cols_to_keep=['recordid', 'so'],
                        papers_query='select recordid, ti, ab, so from semantic.papers;',
                        include_citations=False,
                        citations_query='select * from semantic.citations_avg;'):
    """
    Create features and push to PSQL table.

    Parameters
    ==========
    conn: SQLConn connection
    file_location: string
        location of file to be used to create features 
    features : dict
        Dict of feature names and functions to create those features.
    cols_to_keep : list
        List of strings of columns in papers table to keep in papers_features.
    papers_query : str
        SQL query to pull relevant data to create features on.
    include_citations : boolean
        Whether to include citations data in features table. Defaults to False.
    citations_query: str
        SQL query to pull citations-based features.

    Returns
    =======
    None
    """

    ### Pull papers data ###
    papers = conn.query(papers_query)

    ### Create text features from data ###
    for feature_name, create_feature in tqdm(features.items(), desc='Features'):

        feature = create_feature(papers)
        papers[feature_name] = feature

    ### Keep columns in papers ###
    papers = papers[cols_to_keep + list(features.keys())]

    ### Pull citations-based features, if specified ###
    if include_citations:

        citations_features = conn.query(citations_query)
        papers = papers.merge(citations_features, how='left',
                              left_on='recordid', right_on='c_recordid')
        papers.fillna(0., inplace=True)
        papers.drop(columns='c_recordid', inplace=True)

    ### Pickle dataframe ###
    papers.to_pickle(file_location)


def create_features_production(papers,
                               required_cols=['recordid','ti','ab','so'],
                               features={"tokens_baseline":tokenize_title_abstract_baseline,
                                         "tokens_no_stopwords":tokenize_title_abstract,
                                         "average_embeddings": compute_embeddings,
                                         "stemmed_tokens_nostop": stemming},
                               cols_to_keep=['recordid', 'so'],):
    """
    Computes features for new papers in order to pass them into
    production classifiers to be scored. Borrows functionality from
    create_features_etl(). Returns a pd.DataFrame.

    Parameters
    ==========
    papers : pd.DataFrame
        New papers for which to compute features.
    required_cols : list
        Columns that are required in the csv of new papers.
    features : dict
        Dict of feature names and functions to create those features.
    cols_to_keep : list
        List of strings of columns in papers table to keep in papers_features.

    Returns
    =======
    papers_features : pd.DataFrame
        Dataframe with one row for each paper and
        columns for features needed to score papers.
    """

    ### Check if required columns exist ###
    if not all(col in papers.columns for col in required_cols):
        print("\nRequired columns not present in new papers file.\n")
        return None

    ### Create features from data ###
    for feature_name, create_feature in tqdm(features.items(), desc='Features'):


        feature = create_feature(papers)
        papers[feature_name] = feature

    ### Keep columns ###
    papers = papers[cols_to_keep + list(features.keys())]


    ### Return dataframe ###
    return papers
