import pandas as pd
from scipy import arctan, pi
from sklearn.model_selection import train_test_split

def sample(ignition, connection, local_features_path=None):
    """
    Pulls in dataframe of relevant observations and columns from PSQL.

    Parameters
    ==========
    ignition : yaml with all information necessary
    connection : SQLConn connection class
    local_features_path : str
        Path to locally stored features file.
        If provided, works with features from locally stored file.
        If not provided, works with features stored in PSQL.

    Returns
    =======
    X_train, X_test, y_train, y_test : pd.DataFrames
        X_train, X_test : shape = (# of observations, # of features)
        y_train, y_test : shape = (# of observations, # of classes)
    """

    # pull in all variables of interest from ignition
    # some are no longer use -- may drop some
    e_feature_cols = ignition['existing_features']
    target_col = ignition['target']
    labels_table = ignition['labels_table']
    features_table = ignition['features_table']
    unique_id = ignition['unique_id']
    query = ignition['query']
    data_type = ignition['data_type']
    classes = ignition['classes']
    condition = ignition['condition']
    test_perc = ignition['test_perc']
    seed = ignition['seed']
    sql_seed = (2/pi)*arctan(seed)

    if not unique_id:
        print("You must have a unique id listed to be able to generate test data.")
        return

    if not data_type == "flat":
        print("Data type not supported.")
        return None

    # save required features as string
    ref_features = []

    for e_feature_col in e_feature_cols:
        ref_features.append('semantic.' + features_table + '.' + e_feature_col)

    ref_features = ', '.join(ref_features)

    # condiiton, typically used to limit size of the sample used
    if condition:
        cond = condition
    else:
        cond = ' '


    if local_features_path:
        # get features stored on disk and join to labels from PSQL
        labels_query = f"select setseed({sql_seed}); select * from semantic.{labels_table} {cond};"

        labels_df = connection.query(labels_query)
        labels_df[unique_id] = labels_df[unique_id].astype('int64')

        features_df = pd.read_pickle(local_features_path)
        features_df[unique_id] = features_df[unique_id].astype('int64')

        all_data = labels_df.join(features_df.set_index(unique_id), on=unique_id, how='inner')

    else:
        # get data from SQL database
        query = f"""
            select setseed({sql_seed});
            select {ref_features}, semantic.{labels_table}.* \
            from semantic.{features_table} \
            inner join semantic.{labels_table} \
            on semantic.{features_table}.{unique_id}=semantic.{labels_table}.{unique_id} {cond};"""
        all_data = connection.query(query)

    # split out features (X) and labels (y)
    X = all_data[e_feature_cols]
    labels = [i.lower() for i in classes]
    y = all_data[labels]

    # split data into train and test
    x_train, x_test, y_train, y_test = create_train_test_split(X, y, test_size=test_perc, random_seed=seed)

    return x_train, x_test, y_train, y_test


def create_train_test_split(X, y, test_size=0.2, random_seed=2019):
    """
    Create train test split for data.

    Takes in either a DataFrame or a dictionary of DataFrames, and returns
    train-test splits in the same format (either as DataFrame or as dictionaries
    of DataFrames.)

    Note that the functionality to take in and pass back dictionaries of
    dataframes is no longer used in our ML pipeline.

    Parameters
    ==========
    X : pd.DataFrame or dictionary of pd.DataFrames
        Input data that is to be split in train and test set.
    y : pd.DataFrame or dictionary of pd.DataFrames
        Labels that are to be split in train and test.
    test_size : float
        Proportion of data that should be assigned to test. The other part of the
        data will be assigned to train.
    random_seed : int
        Seed that feeds into the process of selecting random rows for train and test.

    Returns
    =======
    X_train : pd.DataFrame or dictionary of pd.DataFrames
        Train data set(s).
    X_test : DataFrame or dictionary of DataFrames
        Test data set(s).
    y_train : DataFrame or dictionary of DataFrames
        Train labels.
    y_test : DataFrame or dictionary of DataFrames
        Test labels.
    """

    if isinstance(X, pd.DataFrame):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    elif isinstance(X, dict):

        # create empty dictionaries to store files for each class in
        X_train = {}
        X_test = {}
        y_train = {}
        y_test = {}

        for cl in X:

            # create splits for each class
            X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(X[cl], y[cl], test_size=test_size, random_state=random_seed)

            # store in appropriate dictionary
            X_train[cl] = X_cl_train
            X_test[cl] = X_cl_test
            y_train[cl] = y_cl_train
            y_test[cl] = y_cl_test

    return X_train, X_test, y_train, y_test
