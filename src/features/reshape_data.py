def binary_labels(labels, in_class, column="inregister"):

    """
    Converts a multiclass dataset to one with binary labels, where 1 represents
    the "in-class" and 0 represents the "out-classes" (i.e. all classes that are
    not the in-class or the negative examples.)

    Parameters
    ----------
    labels : pd.DataFrame
        All labels across all classes 
        
    in_class : string
        the column for the label of interest
        
    column : string 
        the column to extract which register the papers are in 

    Returns
    -------

        labels : DataFrame
            DataFrame that is identical to that passed to the function - except
            for the column specified as column, where labels have been binarized.

    """

    labels[column] = labels[column].apply(lambda paper: 1 if paper == in_class else 0)

    return labels
