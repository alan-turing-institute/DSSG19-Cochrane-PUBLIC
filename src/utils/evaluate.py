import json
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.metrics import precision_score, recall_score, jaccard_score, confusion_matrix, f1_score, \
    precision_recall_curve
from tqdm import tqdm


def evaluate_accuracy(class_true, class_pred):
    """
    Calculates accuracy given two lists.

    Parameters
    ==========
    class_true : array
        A list of the true class labels.
    class_pred : array
        A list of the predicted class labels.

    Returns
    =======
    float
        Floating point number expressing the accuracy of the predictions.
    """

    accuracy = jaccard_score(class_true, class_pred, average=None)

    return accuracy


def evaluate_recall(class_true, class_pred):
    """
    Calculates recall given two lists.

    For multiclass problems, recall is calculated as a weighted average of the
    recall scores for the individual classes.

    Parameters
    ==========
    class_true : array
        A list of the true class labels.
    class_pred : array
        A list of the predicted class labels.

    Returns
    =======
    float
        Floating point number expressing the recall score of the predictions.
    """

    recall = recall_score(class_true, class_pred, average=None)

    return recall


def evaluate_precision(class_true, class_pred):
    """
    Calculates precision given two lists.

    For multiclass problems, precision is calculated as a weighted average of
    the precision scores for the individual classes.

    Parameters
    ==========
    class_true : array
        A list of the true class labels.
    class_pred : array
        A list of the predicted class labels.

    Returns
    =======
    float
        Floating point number expressing the precision of the predictions.
    """

    precision = precision_score(class_true, class_pred, average=None)

    return precision


def evaluate_confusion_matrix(class_true, class_pred):
    """
    Calculates a confusion matrix.

    A confusion matrix helps to see which classes are hard, and which are
    'confused' by the model.

    Parameters
    ==========
    class_true : array
        A list of the true class labels.
    class_pred : array
        A list of the predicted class labels.

    Returns
    =======
    array, shape = [n_classes, n_classes]
        Floating point number expressing the accuracy of the predictions.
    """

    conf_matrix = confusion_matrix(class_true, class_pred)

    return conf_matrix


def evaluate_f1(class_true, class_pred):
    """
    Calculates f1 score.

    A simple measure of the tests accuracy which considers
    both precision and recall.

    Parameters
    ==========
    class_true : array
        A list of the true class labels.
    class_pred : array
        A list of the predicted class labels.

    Returns
    =======
    float
        Floating point number expressing the precision of the predictions.
    """

    f1 = f1_score(class_true, class_pred, average = None)

    return f1


def evaluate_multilabel_accuracy(class_true, class_pred):
    """
    Compute multi-label accuracy.

    Parameters
    ==========
    class_true : np.ndarray or pd.DataFrame
        True labels, (# of classes)-hot encoded.
    class_pred : np.ndarray or pd.DataFrame
        Predicted labels, (# of classes)-hot encoded.

    Returns
    =======
    accuracies : dict
        Keys for classes and values for accuracy for that class.
    """

    accuracies = ((class_true == class_pred).sum(axis=0)/class_true.shape[0]).fillna('NULL').to_dict()

    return accuracies


def evaluate_multilabel_precision(class_true, class_pred):
    """
    Compute multi-label precision.

    Parameters
    ==========
    class_true : np.ndarray or pd.DataFrame
        True labels, (# of classes)-hot encoded.
    class_pred : np.ndarray or pd.DataFrame
        Predicted labels, (# of classes)-hot encoded.

    Returns
    =======
    precisions : dict
        Keys for classes and values for precisions for that class.
    """

    precisions = (((class_true ==1) & (class_pred == 1)).sum(axis=0)/(class_pred == 1).sum(axis=0)).fillna('NULL').to_dict()

    return precisions


def evaluate_multilabel_recall(class_true, class_pred):
    """
    Compute multi-label recall.

    Parameters
    ==========
    class_true : np.ndarray or pd.DataFrame
        True labels, (# of classes)-hot encoded.
    class_pred : np.ndarray or pd.DataFrame
        Predicted labels, (# of classes)-hot encoded.

    Returns
    =======
    recalls : dict
        Keys for classes and values for recalls for that class.
    """

    recalls = (((class_true ==1) & (class_pred == 1)).sum(axis=0)/(class_true == 1).sum(axis=0)).fillna('NULL').to_dict()

    return recalls


def evaluate_precision_at_k_recall(class_true, class_prob, k):
    """
    Calculates precision for a specified value of recall.

    Parameters
    ==========
    class_true : pd.DataFrame
        True labels, (# of classes)-hot encoded.
    class_prob : pd.DataFrame
        Predicted class probabilities, (# of classes)-hot encoded.
    k : int
        Desired value of recall.

    Returns
    =======
    precisions_at_k_recalls : dict
        Dictionary where keys=classes and values=max precision values.
    """
    ### Checks ###
    if not class_true.shape == class_prob.shape:
        print('class_true and class_prob have different shapes')
        return

    if not set(class_true.columns) == set(class_prob.columns):
        print('class_true and class_prob have different columns')
        return

    if k < 0:
        print("invalid value of recall specified")
        return

    ### Compute ###
    precisions_at_k_recall = {}
    for label in class_true.columns:
        precision, recall, thresholds = precision_recall_curve(class_true[label], class_prob[label])
        max_precision = max(precision[np.where(recall >= k)]) if k != 1 else 0.0
        precisions_at_k_recall[label] = max_precision

    return precisions_at_k_recall


def get_thresholds(y_true, y_pred, minimum_precision=0.95, minimum_recall=0.99):
    """
    Compute thresholds for each review group needed to
    achieve specified minimum_precision and minimum_recall.

    Parameters
    ==========
    y_true : pd. DataFrame
        Dataframe with (at least) columns for the label for
        each paper for each review group.
    y_pred : pd.DataFrame
        Dataframe with (at least) coluns for the prediction for
        each paper for each review group.
    minimum_precision : float
        Minimum value of precision desired for all review groups.
    minimum_recall : float
        Minimum value of recall desired for all review groups.

    Returns
    =======
    upper_thresholds : dict
        Dictionary with keys=review groups and values=upper threshold values.
    lower_thresholds : dict
        Dictionary with keys=review groups and valueslower threshold values.
    """


def get_thresholds(y_true, y_pred, minimum_precision=0.95, minimum_recall=0.99):
    """
        Calculates thresholds given a minimum precision and recall

        Parameters
        ==========
        y_true : pd.DataFrame
            True labels, (# of classes)-hot encoded.
        y_pred : pd.DataFrame
            Predicted class probabilities, (# of classes)-hot encoded.
        minimum_precision : float
            Desired value of precision.
        minimum_recall : float
            Desired value of recall.

        Returns
        =======
        upper lower and recall thresholds : dict
            Dictionary where keys=classes and values=max precision values.
        """
    upper_thresholds = {}
    lower_thresholds = {}
    recall_thresholds = defaultdict(dict)

    for label in tqdm(y_pred.columns, 'Thresholds'):

        precision, recall, thresholds = precision_recall_curve(y_true[label], y_pred[label])

        # upper threshold: all papers with this threshold or higher are classified into RG automatically
        for i, prec in enumerate(precision):
            if prec >= minimum_precision:
                try:
                    upper_thresholds[label] = thresholds[i]
                except:
                    upper_thresholds[label] = 1.0

                recall_thresholds[label]['upper'] = recall[i]

                break

        # lower threshold: all papers below this threshold are discarded automatically
        for i, rec in enumerate(recall):
            if rec < minimum_recall:

                try:
                    lower_thresholds[label] = thresholds[i-1]
                except:
                    lower_thresholds[label] = 0.0

                recall_thresholds[label]['lower'] = recall[i]

                break

    return upper_thresholds, lower_thresholds, recall_thresholds


def get_workload_reduction(y_true, y_pred, upper_thresholds, lower_thresholds):
    """
    Computes the workload reduction breakdown - proportion of papers to keep,
    consider and discard - for each review group based on specified thresholds.

    Parameters
    ==========
    y_true : pd. DataFrame
        Dataframe with (at least) columns for the label for
        each paper for each review group.
    y_pred : pd.DataFrame
        Dataframe with (at least) coluns for the prediction for
        each paper for each review group.
    upper_thresholds : dict
        Dictionary with keys=review groups and values=upper threshold values.
    lower_thresholds : dict
        Dictionary with keys=review groups and valueslower threshold values.

    Returns
    =======
    keep : dict
        Dictionary with key=review group and value=proportion
        recommended to be kept for that review group.
    consider : dict
        Dictionary with key=review group and value=proportion
        recommended to be considered for that review group.
    discard : dict
        Dictionary with key=review group and value=proportion
        recommended to be discarded for that review group.
    """

    keep = {}
    consider = {}
    discard = {}

    for label in y_pred.columns:

        keep_list = [p for p in y_pred[label].tolist() if p >= upper_thresholds[label]]
        consider_list = [p for p in y_pred[label].tolist() if p >= lower_thresholds[label] and p < upper_thresholds[label]]
        discard_list = [p for p in y_pred[label].tolist() if p < lower_thresholds[label]]

        keep_proportion = len(keep_list) / len(y_pred[label].tolist())
        consider_proportion = len(consider_list) / len(y_pred[label].tolist())
        discard_proportion = len(discard_list) / len(y_pred[label].tolist())

        keep[label] = keep_proportion
        consider[label] = consider_proportion
        discard[label] = discard_proportion

    return keep, consider, discard


def select_metric(metric_name, class_true, class_pred, k=None):

    """
    Simple function to pass metric name in and get outcome
    without explicitly calling different functions

    Parameters
    ==========
    metric_name : str
        Name of metric to calculate.
    class_true : array
        A list of the true class labels.
    class_pred : array
        A list of the predicted class labels.
    k : int (optional)
        Desired recall value.

    Returns
    =======
    Computed metric of interest - type varies across metric options.
    See individual metric functions for more detail.
    """

    if metric_name == "accuracy":

        return evaluate_accuracy(class_true, class_pred)

    elif metric_name == "recall":

        return evaluate_recall(class_true, class_pred)

    elif metric_name == "precision":

        return evaluate_precision(class_true, class_pred)

    elif metric_name == "confusion_matrix":

        return evaluate_confusion_matrix(class_true, class_pred)

    elif metric_name == "f1":

        return f1(class_true, class_pred)

    elif metric_name == "multilabel_accuracy":

        return evaluate_multilabel_accuracy(class_true, class_pred)

    elif metric_name == "multilabel_precision":

        return evaluate_multilabel_precision(class_true, class_pred)

    elif metric_name == "multilabel_recall":

        return evaluate_multilabel_recall(class_true, class_pred)

    elif metric_name == "precision_at_recall":

        return evaluate_precision_at_k_recall(class_true=class_true, class_prob=class_pred, k=k)

    else:
        raise ValueError("Name of metric received is not recognized.")


def compute_metrics(metric_names, y_true, y_pred, k=None):
    '''
    Loops through all of specified target classes and computes each
    specified metric for that class and returns dictionary of
    dictionaries. This allows evaluation of a model on a class-by-
    class basis.

    Parameters
    ==========
    metric_names : list
        List of metric_names, in general pulled from ignition.yaml.
    y_true : pd.DataFrame
        For single label case: one column dataframe with true labels for test data.
        For multilabel case: dataframe with one column for each class and a 1 in a cell if that row belongs to that class.
    y_pred : pd.DataFrame
        For single label case: one column dataframe with predicted labels for test data.
        For multilabel case: dataframe with one column for each class and a 1 in a cell if that row is predicted to belong to that class.
    k : int (optional)
        Desired value of recall.

    Returns
    =======
    all_metrics : dict
        Dictionary of dictionaries where primary keys respresent
        each metric and primary values are dictionaries where keys
        are target class and values are metric values.
    '''

    ### ensure that args were provided correctly ###
    if y_true.shape != y_pred.shape:
        print('Features and labels not of same size.')
        return None

    y_true.reset_index(drop=True, inplace=True)
    y_pred.reset_index(drop=True, inplace=True)

    ### compute metrics across metric types and target classes ###
    all_metrics = {}

    for metric in metric_names:

        ### returns dict with key==class and value=metric value ###
        if metric in ['multilabel_accuracy', 'multilabel_precision', 'multilabel_recall']:

            metric_by_class = select_metric(metric, y_true, y_pred)

        elif metric == "precision_at_recall":

            metric_by_class = select_metric(metric, y_true, y_pred, k=k)

        ### returns list of metric values so must iterate over them to put them into dict ###
        else:

            metric_values = select_metric(metric, y_true, y_pred)

            metric_by_class = {}
            for target_class, value in zip(sorted(set(y_true)), metric_values):
                metric_by_class[target_class] = value

        all_metrics[metric] = metric_by_class

    return all_metrics


def results_to_db(metrics, table_name, ignition_id, hash_id, algorithm, hyperparameters, fold, recall, unique_id, connection):
    '''
    Push results of model evaluation to PSQL database.

    Parameters
    ==========
    metrics : dict
        Dictionary of dictionaries where primary keys respresent
        each class and primary values are dictionaries where keys
        are metric names and values are metric values.
    table_name : str
        Name of PSQL table to where results will be pushed.
    ignition_id : str
        ID for iginition file.
    hash_id : str
        ID for ignition file ID + hyperparameters + fold.
    algorithm : str
        Algorithm being used.
    hyperparameters : dict
        Hyperparameters being used.
    fold : str
        Fold number from K-fold cross validation.
    recall : numeric
        Value of recall used to compute precision.
    unique_id : str
        ID for ignition file ID + hyperparameters + fold + recall.
        Should define a unique row in the table.
    connection : active SQLConn class

    Returns
    =======
    None
    '''

    timestamp = str(datetime.now())

    # create schema
    create_schema = "create schema if not exists results;"

    # create table
    create_table = f"""create table if not exists
                    results.{table_name}
                    (unique_id varchar unique, ignition_id varchar, hash_id varchar,
                    algorithm varchar, fold varchar, hyperparameters json, recall numeric,
                    metrics json, timestamp varchar);"""

    # insert into table
    insert_table = f"""
                    insert into results.{table_name}
                    (unique_id,ignition_id,hash_id,algorithm,fold,hyperparameters,recall,metrics,timestamp)
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    on conflict (unique_id)
                    do update
                    set
                        metrics = '{json.dumps(metrics)}',
                        timestamp = '{timestamp}';"""

    # gather sql commands together
    sql_commands = [create_schema, create_table, insert_table]

    # create tuple of ignition_id, hash_id, algorithm and json files (hyperparameters and metrics) to insert
    insert_list = [unique_id, ignition_id, hash_id, algorithm, fold, json.dumps(hyperparameters), recall, json.dumps(metrics), timestamp]
    insert_tuple = tuple(insert_list)

    #execute SQl commands
    for sql_command in sql_commands:
        if "insert" in sql_command:
            connection.execute(sql_command, insert_tuple)
        else:
            connection.execute(sql_command)


def production_results_to_db(table_name, unique_id, review_group, algorithm, hyperparameters, recall,
                            precision, thresholds, recall_at_threshold, workload_reduction, connection):
    '''
    Push results of production model evaluation to database.

    Parameters
    ==========
    table_name : str
        Name of PSQL table to where results will be pushed.
    unique_id : str
        ID for the review group/recall.
    algorithm : str
        Algorithm being used.
    hyperparameters : dict
        Hyperparameters being used.
    recall : float
        Value of recall used to compute precision.
    precision : float
        Value of precision at specified recall.
    thresholds : list
        Upper and lower threshold between which papers should be considered.
    recall_at_threshold : list
        Recall corresponding to upper and lower threshold
    workload_reduction : list
        List storing the workload reduction: proportion that can be automatically kept, should be considered and can be discarded.
    connection : active SQLConn class

    Returns
    =======
    None
    '''

    timestamp = str(datetime.now())

    # create schema
    create_schema = "create schema if not exists results;"

    # create table
    create_table = f"""create table if not exists
                    results.{table_name}
                    (unique_id varchar unique, review_group varchar,
                    algorithm varchar, hyperparameters varchar, recall numeric,
                    precision_at_recall numeric, thresholds varchar, recall_at_threshold varchar, workload_reduction varchar, timestamp varchar);"""

    # insert into table
    insert_table = f"""
                    insert into results.{table_name}
                    (unique_id,review_group,algorithm,hyperparameters,recall,precision_at_recall,thresholds,recall_at_threshold,workload_reduction,timestamp)
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    on conflict (unique_id)
                    do update
                    set
                        precision_at_recall = '{precision}',
                        thresholds = '{thresholds}',
                        hyperparameters = '{json.dumps(hyperparameters).replace("'","")}',
                        algorithm = '{algorithm}',
                        workload_reduction = '{workload_reduction}',
                        recall_at_threshold = '{recall_at_threshold}',
                        timestamp = '{timestamp}';"""

    # gather sql commands together
    sql_commands = [create_schema, create_table, insert_table]

    # create tuple of ignition_id, hash_id, algorithm and json files (hyperparameters and metrics) to insert
    insert_list = [unique_id, review_group, algorithm, json.dumps(hyperparameters).replace("'",""), recall, precision, thresholds, recall_at_threshold, workload_reduction, timestamp]
    insert_tuple = tuple(insert_list)

    #execute SQl commands
    for sql_command in sql_commands:
        if "insert" in sql_command:
            connection.execute(sql_command, insert_tuple)
        else:
            connection.execute(sql_command)
