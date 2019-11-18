### dont need all these -- just for testing purposes
import sys

import pandas as pd

sys.path.append('../utils/')
from load_config import load_psql_env, load_local_paths
from SQLConn import SQLConn
from results import pull_results, get_best_hyperparam_all, get_avg_ignition
import re

from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2)


def plot_optimal_precision_recall_curve(results_df, label):
    """
    Plot the precision recall curve for a specified group.

    Parameters
    ==========

    results_df : DataFrame
        DataFrame containing pipeline_ML results.
    label : string
        Label/review group for which precision-recall curve should be created.

    Returns
    =======

    plt : matplotlib.pyplot object
    """

    # increase font size on plot
    sns.set(font_scale=2)

    # isolate data for review group from dataframe
    visdata = results_df[results_df['label'] == label]

    # create plot
    plt.figure(figsize=(26, 15))
    sns.lineplot(y='precision_at_recall', x='recall', data=visdata, marker="o")
    plt.title(f"Precision vs recall for \n group {label} for hyperparameters {visdata['hyperparameters']}")
    plt.ylabel("Average maximum precision")
    plt.xlabel("Recall")

    return plt


def workload_relative_stackedbar(results_df, sort_by='consider'):
    """
    Plot the relative workload reduction for each group as a stacked
    bar chart.

    Parameters
    ==========

    results_df : DataFrame
        DataFrame containing pipeline_ML results.
    sort_by : string
        Column by which the stacked bars should be sorted.

    Returns
    =======

    plot : matplotlib.pyplot object
    """

    # set image parameters
    sns.set(font_scale=0.5)
    plt.figure(figsize=(52, 30))
    plt.rcParams["figure.dpi"] = 144

    # workload reduction column cleanup - sometimes returns weird format
    if '[' in results_df['workload_reduction'].iloc[0]:
        results_df['workload_reduction'] = results_df['workload_reduction'].apply(lambda x: eval(x))
    else:
        results_df['workload_reduction'] = results_df['workload_reduction'].apply(
            lambda x: re.sub('[{}\']', '', x).split(','))

    # store keep, consider and discard separately
    results_df['keep'] = results_df['workload_reduction'].str[0]
    results_df['keep'] = pd.to_numeric(results_df['keep'])
    results_df['consider'] = results_df['workload_reduction'].str[1]
    results_df['consider'] = pd.to_numeric(results_df['consider'])
    results_df['discard'] = results_df['workload_reduction'].str[2]
    results_df['discard'] = pd.to_numeric(results_df['discard'])

    # sort by specified column (if it exists)
    if sort_by in results_df.columns:
        results_df = results_df.sort_values(by=[sort_by], axis=0)

    # keep only relevant columns
    results = results_df[['review_group', 'keep', 'discard', 'consider']]
    results = results.drop_duplicates()
    print(results)

    # create plot
    plot = results.set_index('review_group').plot(kind='barh', stacked=True, width=1,
                                                  color=['#31a354', '#fb6a4a', '#c6dbef'])

    # set plot parameters
    plt.title("Proportion of papers to keep, discard and consider\n across review groups", fontsize=12)
    plt.ylabel("Review group", fontsize=8)
    plt.xlabel("Proportion of papers", fontsize=8)
    plt.xlim(0, 1)
    plt.legend(ncol=3, loc=3, framealpha=0.95)

    return plot


def plot_average_workload_reduction(results_df):
    """
    Plot the average workload reduction across groups as a stacked
    bar chart.

    Parameters
    ==========

    results_df : DataFrame
        DataFrame containing pipeline_ML results.
    sort_by : string
        Column by which the stacked bars should be sorted.

    Returns
    =======

    plot : matplotlib.pyplot object
    """

    # format data for visualizing
    results_df['keep'] = pd.to_numeric(results_df['workload_reduction'].str[0])
    results_df['consider'] = pd.to_numeric(results_df['workload_reduction'].str[1])
    results_df['discard'] = pd.to_numeric(results_df['workload_reduction'].str[2])

    results_df = results_df[['review_group', 'keep', 'discard', 'consider']]
    results_df = results_df.drop_duplicates()

    # create viz dataframe
    viz_df = pd.DataFrame(
        {'props': [results_df['keep'].mean(), results_df['consider'].mean(), results_df['discard'].mean()]},
        index=['keep', 'consider', 'discard'])

    print(viz_df)
    fig, ax = plt.subplots(facecolor='white')

    # create bar plot
    plot = pd.DataFrame(viz_df['props']).T.plot(kind='barh', stacked=True, color=['#31a354', '#c6dbef', '#fb6a4a'])

    # set plot parameters
    plot.set_facecolor('white')
    plt.title("Average proportion of papers to keep, consider and discard", fontsize=12)
    plt.axis('off')
    plt.legend(ncol=3, loc=8)

    return plot


def plot_precision_recall_curve_best(results_df, label, xlims=(0, 1), best_n=None, best_at=0.95, recall_line=True,
                                     precision_line=True, prec_threshold=0.95,
                                     plot_baseline=True):
    """
    Plot precision recall curves based on different threshold values.

    Parameters
    ==========
    results_df : pd.DataFrame
        Long dataframe with results from specified ignition files, metrics, and labels.
    label : str
        Name of class for which to plot curve.
    xlims : list (optional)
        [x_lower, x_upper] --> use to zoom in on certain areas of plot.
    best_n : int
        Specify the number of best models that have to be plotted. If None, plot all.
    best_at: float
        Threshold at which the models are best.
    recall_line : bool
        Whether a vertical line should be plotted at the specified recall threshold.
    precision_line : bool
        Whether a vertical line should be plotted at a specified precision threshold.
    prec_threshold : float
        Precision treshold at which line should be plotted.
    plot_baseline : bool
        Whether the baseline curve should be included in the plot.

    Returns
    =======
    plt : matplotlib.pyplot object
    """

    sns.set(font_scale=2)

    # isolate the rows for a label
    visdata = results_df[results_df['label'] == label]

    if plot_baseline:
        baseline = visdata[visdata['algorithm'] == 'baseline']
    all_others = visdata[visdata['algorithm'] != 'baseline']

    title = f"Average maximum precision by recall for\ngroup {label}"

    # only take those rows for the best n models
    if best_n is not None:
        # get list of the best_n best models at a certain recall score, and filter those out
        best_models = \
        all_others[all_others['recall'] == best_at].sort_values(by=['precision_at_recall'], ascending=False).head(
            best_n)['algorithm'].tolist()
        all_others = all_others[all_others['algorithm'].isin(best_models)]
        title = f"{title} - best {best_n} algorithm(s)"

    plt.figure(figsize=(26, 15))

    palette = sns.color_palette('colorblind', all_others['algorithm'].nunique())

    # plot curve(s)
    ax = sns.lineplot(y='precision_at_recall', x='recall', hue='algorithm', data=all_others, marker="o",
                      palette=palette)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[2:], labels=labels[2:])

    # include baseline in plot
    if plot_baseline:
        ax2 = sns.lineplot(y='precision_at_recall', x='recall', data=baseline, hue='algorithm', marker="o", lw=3,
                           palette={'baseline': '#d62728'})

        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles2[1:], labels=labels2[1:])

    # include recall threshold line
    if recall_line is not None:

        # get position of line either from data or directly
        if not 'recall_at_threshold' in all_others.columns:
            xline = best_at
        else:
            xline = max(eval(all_others.head(1).reset_index()['recall_at_threshold'][0]))

        plt.axvline(x=xline, color='r')
        ax.axvspan(xline, 1, alpha=0.15, color='red')

    # include precision threshold line
    if precision_line is not None:

        if not 'recall_at_threshold' in all_others.columns:

            best_models = \
            all_others[all_others['recall'] == best_at].sort_values(by=['precision_at_recall'], ascending=False).head(
                1)['algorithm'].tolist()
            all_others = all_others[all_others['algorithm'].isin(best_models)]

            precision_list = all_others['precision_at_recall'].tolist()
            recall_list = all_others['recall'].tolist()

            for i, prec in enumerate(precision_list):
                if prec < prec_threshold:
                    i_upper = i
                    i_lower = i - 1
                    break

            # calculate where line should be drawn
            slope = (precision_list[i_upper] - precision_list[i_lower]) - (recall_list[i_upper] - recall_list[i_lower])
            intercept = precision_list[i_lower] - slope * recall_list[i_lower]
            slope, intercept, _, _, _ = linregress([recall_list[i_lower], recall_list[i_upper]],
                                                   [precision_list[i_lower], precision_list[i_upper]])
            xline2 = prec_threshold / slope - intercept / slope

        else:
            xline2 = min(eval(all_others.head(1).reset_index()['recall_at_threshold'][0]))

        plt.axvline(x=xline2, color='r')
        ax.axvspan(0, xline2, alpha=0.15, color='green')
        ax.axvspan(xline2, xline, alpha=0.15, color='orange')

    # plot parameters
    plt.title(title)
    plt.ylabel("Average maximum precision")
    plt.xlabel("Recall")
    if xlims:
        plt.xlim(xlims)

    return plt


def plot_precision_recall_curve_hyperparams(results_df, label, xlims=None):
    """
    Plot precision recall curves based on different threshold values.

    Parameters
    ==========
    results_df : pd.DataFrame
        Long dataframe with results from specified ignition files, metrics, and labels.
    label : str
        Name of class for which to plot curve.
    xlims : list (optional)
        [x_lower, x_upper] --> use to zoom in on certain areas of plot.

    Returns
    =======
    None
    """

    sns.set(font_scale=2)

    # select data
    results_df['hyperparameters'] = results_df['hyperparameters'].astype('str')
    visdata = results_df[results_df['label'] == label][['hyperparameters', 'recall', 'precision_at_recall']].groupby(
        ['hyperparameters', 'recall']).mean().reset_index()

    # make plot
    plt.figure(figsize=(26, 15))
    sns.lineplot(y='precision_at_recall', x='recall', hue='hyperparameters', data=visdata, marker="o", legend=False)
    plt.title(f"Average maximum precision by recall for\ngroup {label}")
    plt.ylabel("Average maximum precision")
    plt.xlabel("Recall")
    if xlims:
        plt.xlim(xlims)

    return plt


def plot_distribution_precision(results_df, recall):
    """
    Plot distribution of precisions at a given recall (as a bar chart).

    Parameters
    ==========

    results_df : DataFrame
        Dataframe containing results for multiple models.
    recall : float
        Recall value to be measured.

    Returns
    =======

    plot : matplotlib.pyplot object
    
    """

    sns.set(font_scale=1)

    plot = results_df[results_df['recall'] == recall]['precision_at_recall'].hist(bins=15)

    title_plot = f"Distribution of precision scores across RGs for a recall of {recall}"
    plt.title(title_plot)
    plt.xlabel("Precision scores")
    plt.ylabel("Number of RGs")

    return plot


def plot_workload_reductions(workload_reductions):
    """
    Plot workload reductions across groups (as histogram).

    Parameters
    ==========

    workload_reductions : dict
        Dictionary with review groups as keys and workload reductions
        as values.

    Returns
    =======

        plot : matplotlib.pyplot object
    """

    df = pd.DataFrame.from_dict(workload_reductions, orient='index', columns=['reduction'])
    plot = df['reduction'].hist(bins=15)
    plt.title("Distribution of reduced workload across groups")
    plt.xlabel("Reduced workload in %")
    plt.ylabel("Number of RGs")

    return plot


def heatmap_data(igs, comp_ig='1', local_paths='../pipeline/local_paths.yaml'):
    """
    
    Function that prepares data to be plotted in heatmap, and is stored as csv.

    Parameters
    ==========
    igs : list
        List of strings of ignition ids to be pulled from database.
    
    """

    out_dir = load_local_paths(local_paths)['tmp']

    # pull in results data
    results = pull_results(igs)
    best = get_best_hyperparam_all(results)
    best = best[['algorithm', 'hyperparameters', 'label', 'recall', 'precision_at_recall']]
    best['type'] = 'best'
    comp = get_avg_ignition(results, comp_ig)
    comp['type'] = 'comp'
    out = pd.concat([comp, best], axis=0)

    # pull rg data
    connection = SQLConn(load_psql_env(load_local_paths(local_paths)['pgpass_path']))
    connection.open()
    papers_rgs = connection.query(
        'select inregister as label, count(*) as n_papers from semantic.papers_rgs group by 1;')
    papers_revs = connection.query(f"""
                    with tbl as(select a.*, b.cn from semantic.papers_rgs a
                    left join semantic.papers_reviews b on a.recordid=b.recordid)
                    select inregister as label, count(distinct cn) as n_revs from tbl group by 1;
                    """)

    # final dataset
    out = pd.merge(out, papers_rgs, 'left', 'label')
    out = pd.merge(out, papers_revs, 'left', 'label')

    # output dataset
    out.to_csv(out_dir + 'heatmap_data.csv')
