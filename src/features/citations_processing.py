import pandas as pd

import sys
sys.path.append('..')
sys.path.append('../utils/')

from tqdm import tqdm


def recordid_to_paperid(conn, recordid, lookup_table='semantic.recordid_paperid'):
    """
    Does a look up for the paperid corresponding to a recordid. If a unique link,
    exists, returns that paperid, otherwise returns nothing.

    Parameters
    ==========
    conn : SQLConn conection
    recordid : str
    lookup_table : str

    Returns
    =======
    paperid : str
    """

    query = f"select paperid from {lookup_table} where recordid='{recordid}';"
    paperid = conn.query(query)

    if paperid.shape[0] == 1:
        return paperid['paperid'][0]
    else:
        return None


def get_cited_paperids(conn, paperid,
                       original_paper_col='paperid', cite_col='refpaperid',
                       citations_table='semantic.citations'):
    """
    Get all of the paperids cited by one paperid. If cited papers exist, returns
    a list of those paperids, otherwise returns None.

    Parameters
    ==========
    conn : SQLConn connection
    paperid : str
    original_paper_col : str
    cite_col : str
    citations_table : str

    Returns
    =======
    cited_paperids : list
    """

    query = f"select {cite_col} from {citations_table} where {original_paper_col}='{paperid}';"
    cited_paperids = conn.query(query)

    if cited_paperids.shape[0] > 0:
        return cited_paperids[cite_col].to_list()
    else:
        return None


def create_citations_features_rgs(papers, conn, col_fun='mean',
                                  labels_table_name='semantic.papers_rgs_wide_mod'):
    """
    For each papers in papers, creates a vector of the proportion of that paper's
    cited papers that ended up in each of the 54 review groups in our dataset.

    Parameters
    ==========
    papers : pd.DataFrame
        Papers for which to compute this citation-based feature.
    conn : SQLConn connection
    col_fun : str
        Function to use to aggregate cited papers labels.
        Accepted values are ['mean', 'sum'].
    labels_table_name : str
        Name of PSQL table to use to extract information about a paper's labels.

    Returns
    =======
    citations_features : list of lists
        List of lists of length 54, corresponding to the proportion of
        that paper's cited papers that were labeled into each review group.
    """

    ### Pull in labels table ###
    query = f"select * from {labels_table_name};"
    labels = conn.query(query)

    ### Drop duplicate external IDs ###
    # temporary fix
    labels = labels.drop_duplicates(subset='paperid', keep=False)

    # loop over all recordids in papers table and compute new citations feature
    citation_labels = []
    for recordid in tqdm(papers['recordid'], desc='Cited papers labels'):

        # get paperid
        paperid = recordid_to_paperid(conn=conn, recordid=recordid)

        # check if there is a paperid for that recordid
        if paperid is not None:

            # get cited paperids
            cited_paperids = get_cited_paperids(conn=conn, paperid=paperid)

            # check if there are cited papers
            if cited_paperids is not None:

                # get a dataframe of just the labels for the cited papers
                cited_labels = labels.loc[labels['paperid'].isin(cited_paperids)]

                # check if any of these cited papers have labels
                if cited_labels.shape[0] > 0:

                    # aggregate labels of these cited papers
                    if col_fun == "mean":
                        aggregated_cited_labels = cited_labels.mean(axis=0)
                    elif col_fun == "sum":
                        aggregated_cited_labels = cited_labels.sum(axis=0)
                    else:
                        print("col_fun not supported")
                        return

                    # remove the recordid and paperid
                    aggregated_cited_labels_list = aggregated_cited_labels.to_list()[2:] # remove recordid and paperid

                    # append to list of features
                    citation_labels.append(aggregated_cited_labels_list)

                else:
                    # append a list of 0s of length = # review groups
                    citation_labels.append([0]*54)

            else:
                # append a list of 0s of length = # review groups
                citation_labels.append([0]*54)

        else:
            # append a list of 0s of length = # review groups
            citation_labels.append([0]*54)

    return citation_labels
