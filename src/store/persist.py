import pickle
import os.path
import pandas as pd
import hashlib


def save_pickle(file, location, filename):
    """
    Save pickled model object to disk.

    Parameters
    ==========
    file : scikit-learn/custom model object or dict
        File to pickle and store on disk.
    location : str
        Path to directory to store the pickled model object.
    filename : str
        Name of file to be stored.


    Returns
    =======
    None
    """
    with open(f"{location}/{filename}.pkl", 'wb') as f:
        pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_csv(df, location, filename):
    """
    Save compressed csv to disk.

    Parameters
    ==========
    df : pd.DataFrame
        Dataframe to be stored on disk.
    location : str
        Path to directory to store the file.
    filename : str
        Name of file to be stored.

    Returns
    =======
    None
    """
    df.to_csv(path_or_buf=f"{location}/{filename}.csv.gz", compression='gzip', index=False)


def save(object, location, filename, persist=True):
    """
    Save wrapper function. Check which file type should be saved and routes
    to the correct function for saving.

    Parameters
    ==========
    object :
        Object to be stored.
    location : str
        Path to directory where file is stored on disk.
    filename : str
        Name of file to be saved.

    Returns
    =======
    None
    """

    if persist:
        if isinstance(object, pd.DataFrame):
            save_csv(object, location, filename)
        else:
            save_pickle(object, location, filename)


def load_pickle(location, filename):
    """
    Load pickled object from disk.

    Parameters
    ==========
    location : str
        Path to directory to store the pickled object.
    filename : str
        Name of file to be loaded.

    Returns
    =======
    object : scikit-learn/custom model object or dictionary
        Object loaded from disk.
    """
    with open(f"{location}/{filename}.pkl", 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_csv(location, filename):
    """
    Load compressed csv from disk.

    Parameters
    ==========
    location : str
        Path to directory where file is stored on disk.
    filename : str
        Name of file to be loaded.

    Returns
    =======
    df : pd.DataFrame
        Dataframe loaded from disk.
    """
    df = pd.read_csv(filepath_or_buffer=f"{location}/{filename}.csv.gz", compression='gzip')
    return df


def load(location, filename):
    """
    Load wrapper function. Check which file type should be loaded and routes
    to the correct function for loading.

    Parameters
    ==========
    location : str
        Path to directory where file is stored on disk.
    filename : str
        Name of file to be loaded.

    Returns
    =======
    file :
        File that is loaded.
    """

    if os.path.isfile(f"{location}/{filename}.pkl"):
        return load_pickle(location, filename)

    elif os.path.isfile(f"{location}/{filename}.csv.gz"):
        return load_csv(location, filename)

    else:
        print("File not found!")
        return None


def create_hash_id(input):
    """
    Creates hash identifier.

    Parameters
    ==========
    input : str
        Input to be hashed to form an identifier.

    Returns
    =======
    hash_id : str
        Identifier formed by hashing the inputs.
    """
    input_bytes = bytes(input, encoding="utf-8")
    hash_id = hashlib.sha224(input_bytes).hexdigest()
    return hash_id


def check_persisted(location, filename, load_fresh):
    """
    Check if a file is already stored on disk.

    Parameters
    ==========
    location : str
        Path to directory where file is stored on disk.
    filename : str
        Identifier corresponding to file of interest.
    load_fresh: bool
        If true, do not load from file

    Returns
    =======
    persisted : boolean
        If true, the file exists on disk in the specified location.
    """

    if load_fresh:
        return False

    possible_extensions = ['pkl', 'csv.gz']

    for extension in possible_extensions:
        if os.path.isfile(f"{location}/{filename}.{extension}"):
            return True

    return False
