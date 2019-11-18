import subprocess
import sys
import sqlalchemy

sys.path.append('../utils/')
sys.path.append('../features/')
from load_config import load_psql_env, load_local_paths
from extract_data_from_reviews import create_reviews_tables
from etl_features import create_features_etl
from etl_pipe_tools import papers_long_to_wide


def run_etl(starting_point, env, papers=None, reviews_dir=None, citations_dict=None):
    """
    Runs ETL process from varying starting points and pushes data to PSQL database.

    Parameters
    ==========
    starting_point : str
        Where to start ETL process. Possible values are 'no_sql', 'raw', 'clean'.
    env : dict
        Dictionary with PSQL environment variables.
    reviews_dir : str
        OPTIONAL: Path to directory with review files.
    citations_dict : dict
        OPTIONAL: Dictionary with keys = citations table
        names and values = paths to files to copy to tables.

    Returns
    =======
    None
    """

    if starting_point == 'no_sql':
        no_sql_to_raw(reviews_dir=reviews_dir, citations_dict=citations_dict, env=env)
        raw_to_clean(env)
        clean_to_semantic(env)

    elif starting_point == 'raw':
        raw_to_clean(env)
        clean_to_semantic(env)

    elif starting_point == 'clean':
        clean_to_semantic(env)

    else:
        print('Not a valid starting point.')


def no_sql_to_raw(reviews_dir, citations_dict, env):
    """
    Extracts data from local files and pushes data to raw PSQL tables.

    Parameters
    ==========
    reviews_dir : str
        Path to directory with review files.
    citations_dict : dict
        Dictionary with keys = citations table names
        and values = paths to files to copy to tables.
    env : dict
        Dictionary with PSQL environment variables.

    Returns
    =======
    None
    """
    DIR = '../../sql/raw/'

    # 0. Open database connection
    db_url = sqlalchemy.engine.url.URL(drivername='postgresql', username=env['user'], password=env['password'], host='localhost', database=env['dbname'])
    engine = sqlalchemy.create_engine(db_url)
    conn = engine.raw_connection()
    cur = conn.cursor()

    # 1. Create raw schema
    while True:
        create_raw_schema = input("Do you want to re-create raw schema? WARNING: this will delete the papers table, which takes a long time to build. [y/n]")
        if not create_raw_schema.lower() in ['y','n']:
            print("Please provide a valid option.")
            continue
        else:
            break

    if create_raw_schema == 'y':
        with open(DIR + 'create_schema_raw.sql', 'r') as f:
            cur.execute(f.read())
        print('raw schema created')

    elif create_raw_schema == 'n':
        pass

    # 2. Extract papers data from papers and push to raw table
    while True:
        create_raw_papers = input("Do you want to re-build raw.papers? WARNING: this may take a long time. [y/n]")
        if not create_raw_schema.lower() in ['y','n']:
            print("Please provide a valid option.")
            continue
        else:
            break

    if create_raw_papers == 'y':
        print('raw.papers table created')
        retcode = subprocess.call("Rscript ../../burning_bus/keep_paper_return_issue.R", shell=True)
        print('raw.papers table populated')
    elif create_raw_papers == 'n':
        pass

    # 3. Extract data from reviews and push to raw tables
    reviews_df, reviews_studies_df = create_reviews_tables(reviews_dir)
    print('reviews_df and reviews_studies_df created')

    # reviews table
    with open(DIR + 'create_reviews.sql', 'r') as f:
        cur.execute(f.read())
    print('raw.reviews table created')

    reviews_df.to_sql(table='reviews', con=conn, schema='raw')
    print('raw.reviews table populated.')

    # reviews_studies table
    with open(DIR + 'create_reviews_studies.sql', 'r') as f:
        cur.execute(f.read())
    print('raw.reviews_studies table created')

    reviews_studies_df.to_sql(table='reviews_studies', con=conn, schema='raw')
    print('raw.reviews_studies table populated')

    # 4. Push citations data to raw tables.
    # create
    with open(DIR + 'create_citations.sql', 'r') as f:
        cur.execute(f.read())
    print('raw.citations and raw.recordid_paperid tables created')

    # # copy from local
    for table_name, path in citation_dict.items():
        with open(path, 'r') as f:
            next(f) # skip header
            cur.copy_from(f, table_name, sep=',')
    print('raw.citations and raw.recordid_paperid tables populated')

    conn.commit()
    cur.close()
    conn.close()


def raw_to_clean(env):
    """
    Runs scripts to move data from raw tables to cleaned tables in PSQL.

    Parameters
    ==========
    env : dict
        Dictionary with PSQL environment variables.

    Returns
    =======
    None
    """
    DIR = '../../sql/clean/'

    # 0. Open database connection
    db_url = sqlalchemy.engine.url.URL(drivername='postgresql', username=env['user'], password=env['password'], host='localhost', database=env['dbname'])
    engine = sqlalchemy.create_engine(db_url)
    conn = engine.raw_connection()
    cur = conn.cursor()

    ### 1. Create schema ###
    with open(DIR + 'create_clean_schema.sql', 'r') as f:
        cur.execute(f.read())
    print('clean schema created')

    ### 2. Run scripts for papers ###
    with open(DIR + 'create_papers_clean.sql', 'r') as f:
        cur.execute(f.read())
    print('clean.papers table created and populated')

    ### 3. Run scripts for reviews ###
    with open(DIR + 'create_reviews_clean.sql', 'r') as f:
        cur.execute(f.read())
    print('clean.reviews table created and populated')

    ### 4. Run scripts for citations ###
    with open(DIR + 'create_citations_clean.sql', 'r') as f:
        cur.execute(f.read())
    print('clean.citations and clean.recordid_paperid tables created and populated')

    conn.commit()
    cur.close()
    conn.close()


def clean_to_semantic(env):
    """
    Runs scripts to move data from clean tables to semantic tables in PSQL.

    Parameters
    ==========
    env : dict
        Dictionary with PSQL environment variables.

    Returns
    =======
    None
    """
    DIR = '../../sql/semantic/'
    # 0. Open database connection
    db_url = sqlalchemy.engine.url.URL(drivername='postgresql', username=env['user'], password=env['password'], host='localhost', database=env['dbname'])
    engine = sqlalchemy.create_engine(db_url)
    conn = engine.raw_connection()
    cur = conn.cursor()
    start_time = datetime.datetime.now()
    print("End time is " + str(start_time))
    ### 1. Create schema ###
    with open(DIR + 'create_semantic_schema.sql', 'r') as f:
        cur.execute(f.read())
    print('semantic schema created')

    ### 2. Run scripts for papers ###
    with open(DIR + 'create_papers_semantic.sql', 'r') as f:
        cur.execute(f.read())
    print('semantic.papers table created and populated')

    with open(DIR + 'create_papers_revs_semantic.sql', 'r') as f:
        cur.execute(f.read())
    print('semantic.papers_revs table created and populated ')

    with open(DIR + 'create_papers_rgs_semantic.sql', 'r') as f:
        cur.execute(f.read())
    print('semantic.papers_rgs table created and populated')

    ### 3. Run scripts for reviews ###
    with open(DIR + 'create_reviews_semantic.sql', 'r') as f:
        cur.execute(f.read())
    print('semantic.reviews created and populated')

    ### 4. Run scripts for citations ###
    with open(DIR + 'create_citations_semantic.sql', 'r') as f:
        cur.execute(f.read())
    print('semantic.citations and semantic.recordid_paperid tables created and populated')

    ### 5. Run .py scripts to create features ###
    # flag include_citations=True for citations features
    create_features_etl(conn, cur, include_citations=False)
    print('semantic.features table created and populated')

    ### 6. Run scripts for features table creation ###
    papers_long_to_wide(conn)
    print('semantic.papers_rgs_wide table created and populated')

    ### 7. Run script to compute citations-based features
    with open(DIR + 'create_citations_avg_semantic.sql', 'r') as f:
        cur.execute(f.read())
    print('semantic.citations_avg created and populated')

    conn.commit()
    cur.close()
    conn.close()
    end_time = datetime.datetime.now()
    print("End time is " + str(end_time))
    print("Elapsed time is " + str(end_time-start_time))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Please specify one (and only one) argument for starting_point.')
    else:
        starting_point = sys.argv[1]
        dot_env = load_local_paths('../pipeline/local_paths.yaml')
        env = load_psql_env(pgpass_path=dot_env['pgpass_path'])
        print(env)

        if starting_point == 'no_sql':
            reviews_dir = '/data/raw/reviews/'
            citations_dict = {'citations': '/data/citations/TuringCitations.csv','recordid_paperid':'/data/citations/TuringCRSPMRecords.csv'}
            run_etl(starting_point=starting_point, env=env, reviews_dir=reviews_dir, citations_dict=citations_dict)
        else:
            run_etl(starting_point=starting_point, env=env)
