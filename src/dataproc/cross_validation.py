import numpy as np
import pandas as pd


def k_fold(X, Y, k, seed=100):
    """
    Inputs
    ------
    X : dataframe or dictionary of dataframes which are features

    Y : dataframe or dictionary of dataframe which are labels

    k : int
        Number of folds

    seed : int
        set seed retrieved from ignition yaml

    Outputs
    -------
    List of tables or k lists of dictionaries of tables for
    X and Y

    """

    #initiate kfolds
    np.random.seed(seed)

    if isinstance(X, pd.DataFrame):

        random_ids = np.random.randint(0, k, X.shape[0])
        X['k'] = random_ids
        Y['k'] = random_ids

        return X, Y

    elif isinstance(X, dict):

        #error checking to ensure that the number of keys in Y and X
        #are the same
        if(len(X.keys())==len(Y.keys())):

            #iterate through each key in the dictionary
            for key in X:

                #create generator
                splits = kf.split(X[key])
                split = next(splits, None)
                i=1

                #generate empty lists for the kfolds within the dict
                X[key]['k'] = 0
                Y[key]['k'] = 0

                while(split):
                    X[key].iloc[split[1],'k'] = i
                    Y[key].iloc[split[1],'k'] = i
                    i += 1
                    split = next(splits, None)
            return X, Y
        else:
            print("The number of keys in X and Y are not equal")


    else:
        print("The data must be a dictionary of pd.dfs or a pd.df")
