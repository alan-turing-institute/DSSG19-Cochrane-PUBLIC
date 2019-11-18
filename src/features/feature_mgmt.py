from text_processing import create_vocabularies


def create_fold_feature(fold_x_train, fold_y_train, feature):

    """
    Create one feature (for k-1 folds during k-fold cross validation) as required for a specific model.

    After pulling data from the database, we may wish to create additional features.
    This function currently supports creation of vocabularies.

    Parameters
    ----------

    fold_x_train : pd.DataFrame or dictionary containing DataFrames.
        The input training data for k-1 folds over which features should be calculated.

    fold_y_train : pd.DataFrame
        The input training labels for k-1 folds to be used to subset fold_X_train for certain feature creation.

    feature : str
        Name of the feature that this function should calculate.

    Returns
    -------

    computed_feature : dependent on which feature is created

    """

    # create vocabulary (i.e., for baseline model from James)
    if feature == 'vocab':

        vocabularies = create_vocabularies(fold_x_train, fold_y_train)

        print("Created vocabularies.")

        return vocabularies

    else:

        print('No other features are currently supported')

        return None
