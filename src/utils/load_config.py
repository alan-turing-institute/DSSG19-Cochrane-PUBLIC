import yaml


def load_psql_env(pgpass_path):
    """
    Loads PSQL environment variables.

    Parameters
    ==========
    pgpass_path : str
        Local path to .pgass file.

    Returns
    =======
    env : dict
        host : str
        port : str
        dbname: str
        user: str
        password: str

    """

    with open(pgpass_path,'r') as f:
        psql_info = [line for line in f if 'cochrane' in line][0]

    host, port, dbname, user, password = psql_info.replace('\n','').split(':')

    env = {'host':host,'port':port,'dbname':dbname,'user':user,'password':password}

    return env


def load_config(file, append_static=True, static = "../config/static.yaml"):

    """

    Load a YAML file with specifications for one ML pipeline run.

    Parameters
    ==========
    file : str
        Local path to .yaml file.
    append_static : boolean
        Whether to append static.yaml. Defaults to True.
    static : str
        Local path to static.yaml file.

    Returns
    =======
    ignition : dict
        Specifications for one model run.
    """

    if append_static:
        with open(static, 'r') as f:
            try:
                static = yaml.safe_load(f)
            except yaml.YAMLError as err:
                print(err)
                return None

    with open(file, 'r') as f:
        try:
            ignition = yaml.safe_load(f)
            if append_static:
                ignition.update(static)
        except yaml.YAMLError as err:
            print(err)
            return None

    return ignition


def load_local_paths(path):
    """
    Load local paths .yaml file.

    Parameters
    ==========
    path : str
        Path to .env file.

    Returns
    =======
    local_paths : dict
        Dictionary with environmental variables from .yaml file.
    """

    with open(path, 'r') as f:
        try:
            local_paths = yaml.safe_load(f)
        except yaml.YAMLError as err:
            print(err)
            return None

    return local_paths
