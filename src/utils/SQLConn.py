import pandas as pd
import psycopg2
import io


class SQLConn:

    def __init__(self, env):
        if len(env) != 5:
            print("env should be of length 5")
        else:
            self.dbname = env['dbname']
            self.user = env['user']
            self.host = 'localhost'
            self.password = env['password']
            self.envinfo = True
            self.isopen = False
            self.conn = None
            self.cursor = None

    def open(self):
        if self.envinfo:
            try:
                self.conn = psycopg2.connect(dbname=self.dbname, user=self.user, host=self.host, password=self.password)
                self.cursor = self.conn.cursor()
                self.isopen = True
            except(Exception, psycopg2.Error) as error:
                print('Error while connecting to PostgreSQL Server', error)
        else:
            print("There is no environment file provided, please run 'SQLConn(env)' to initialize'")

    def execute(self, query, insert_tuple=None):
        """
        Executes query directly in PSQL.

        Parameters
        ==========
        query : str
            SQL query.
        insrt_tuple : tuple (optional)
            Tuples of row values to insert into a PSQL table.
        """

        # refresh instance
        self.close()
        self.open()

        if insert_tuple:
            self.cursor.execute(query, insert_tuple)

        else:
            self.cursor.execute(query)

        self.conn.commit()

        self.close()

    def query(self, query):
        """
        Takes in the query and returns a pd.DataFrame.

        Parameters
        ==========
        query : str
            Sql query.

        Returns
        =======
        df : pd.DataFrame
        """

        if self.isopen:
            try:
                self.conn = psycopg2.connect(dbname=self.dbname, user=self.user, host=self.host, password=self.password)
                self.isopen = True
                sample = pd.read_sql_query(query, self.conn)
                return sample
            except(Exception, psycopg2.Error) as error:
                print('Error while connecting to PostgreSQL Server', error)
        else:
            print("You do not have a connection open")

    def fastpush(self, table_name, pd_df, var_type):
        """
        Takes in a table_name, pd.DataFrame and pushes to table in PSQL
        using StringIO(). Will drop table if it exists and re-create it.

        Parameters
        ==========
        table_name: str
            Name of SQL table to push to.
        pd_df : pd.DataFrame
            Dataframe to push to that table.
        var_type : str
            Data type for SQL table.

        Returns
        =======
        None
        """

        # refresh instance
        self.close()
        self.open()

        if isinstance(table_name, str) and isinstance(pd_df, pd.DataFrame) and self.isopen:
            col_sql_create = ', '.join(list(map(lambda x: x + ' ' + var_type, pd_df.columns)))
            col_sql_push = ', '.join(pd_df.columns)
            try:
                f = io.StringIO()
                pd_df.to_csv(f, index=False, header=False)
                f.seek(0)  # move position to beginning of file before reading
                self.cursor.execute(f"""drop table if exists {table_name};
                                create table {table_name} ({col_sql_create});""")
                copy_table = f'copy {table_name} ({col_sql_push}) from stdin with csv'
                self.cursor.copy_expert(sql=copy_table, file=f)  # f : StringIO object
                self.conn.commit()

            except(Exception, psycopg2.Error) as error:
                print('Error while pushing to PostgreSQL Server', error)
        else:
            print("SQL connection must be open, table name must be a string and pd_df must be a pandas dataframe")

    def close(self):
        """
        Closes pyscopg2 connection.
        """
        if self.isopen:
            self.cursor.close()
            self.conn.close()
            self.isopen = False
