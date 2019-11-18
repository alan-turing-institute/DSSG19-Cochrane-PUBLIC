import sys

sys.path.append('../utils/')
from load_config import load_psql_env, load_local_paths
from SQLConn import SQLConn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pandas as pd


def plot_citations_histograms(data, review_group):
    """
    Plot two histograms on the same plot - one for the distribution of
    the citation features for a review group for papers belonging to that
    review group and another for those not belonging to that review group.

    Parameters
    ==========
    data : pd.DataFrame
        Required columns: [f'{review_group}', f'cited_{review_group}'].
    review_group : str
        Name of review group (also name of the column).

    Returns
    =======
    None
    """
    color_in = "#1f497d"
    color_out = "firebrick"

    plt.hist(data.loc[data[review_group] == True][f'cited_{review_group}'],
             color=color_in, alpha=0.5, density=True)
    plt.hist(data.loc[data[review_group] == False][f'cited_{review_group}'],
             color=color_out, alpha=0.5, density=True)
    plt.title(review_group, fontsize=20);


def plot_citations_features_small_multiples(conn, save_path='/data/figs/eda/citations_histograms.png'):
    """
    Creates a small multiples plot with of plot_citations_histograms().
    Saves file to save_path.

    Parameters
    ==========
    conn : SQLConn class instance
    save_path : str
        Local path where plot should be saved.

    """

    ### Pull data ###
    query = f"""
    select a.*, b.*
    from semantic.papers_rgs_wide a
    left join semantic.citations_avg b
    on a.recordid=b.c_recordid;
    """

    data = conn.query(query)
    data.fillna(0, inplace=True)

    ### Pull labels ###
    bad_cols = ['recordid', 'c_recordid', 'citations_available']
    review_groups = []
    for col in list(data.columns):
        if not col in bad_cols and col[:5] != "cited":
            review_groups.append(col)

    ### Prep plot legend ###
    red_patch = mpatches.Patch(color='firebrick', label='Non-review group paper')
    blue_patch = mpatches.Patch(color='#1f497d', label='Review group paper')

    ### Plot ###
    fig = plt.figure(figsize=(40, 40))
    for i, review_group in enumerate(review_groups):
        plt.subplot(8, 8, i + 1)
        plot_citations_histograms(data=data, review_group=review_group)
    # plt.suptitle('Distribution of proportion of cited papers belonging to that review group', fontsize=50, y=1.05)
    plt.figlegend(handles=[red_patch, blue_patch], loc='lower right', fontsize=30)
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    connection = SQLConn(load_psql_env(load_local_paths('../pipeline/local_paths.yaml')['pgpass_path']))
    connection.open()
    plot_citations_features_small_multiples(conn=connection)
    connection.close()
