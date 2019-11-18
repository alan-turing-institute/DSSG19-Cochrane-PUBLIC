import sys
sys.path.append('../utils/')


def papers_long_to_wide(connection):
    """
    Converts the review group labels from long to wide format and
    pushes a new table to the PSQL database. Default name of long table
    is `semantic.papers_rgs` and default name of new table is
    `semantic.papers_rgs_wide`.

    Parameters
    ==========
    connection : active SQLConn class instance

    Returns
    =======
    None
    """

    long = connection.query('select * from semantic.papers_rgs')

    # add index column
    long['label'] = 1

    # pivot, and add zeroes on the empty cells
    wide = long.pivot(index='recordid', columns='inregister', values='label').fillna(0)
    wide = wide.reset_index()
    
    connection.fastpush('semantic.papers_rgs_wide', wide)
