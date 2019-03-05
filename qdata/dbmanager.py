import pyodbc
import pandas as pd
from pandas.io.sql import read_sql
from qinv.settings import sql_config

__version__ = '1.0.1'


class SqlManager:
    def __init__(self, server_name=sql_config['server_name']
                 , db_name=sql_config['db_name']
                 , usr=sql_config['usr']
                 , pwd=sql_config['pwd']):
        self.conn = None
        self.server_name = server_name
        self.db_name = db_name
        self.usr = usr
        self.pwd = pwd

    def get_usr_info(self):
        return {'server_name': self.server_name, 'db_name': self.db_name, 'usr': self.usr}

    def set_usr_info(self, usr, pwd, db_name=sql_config['db_name']):
        self.usr = usr
        self.pwd = pwd
        self.db_name = db_name

    def set_db_name(self, db_name):
        self.db_name = db_name

    def get_conn_str(self):
        return r'DRIVER={{SQL Server}};SERVER={0};DATABASE={1};UID={2};PWD={3}'.format(
            self.server_name, self.db_name, self.usr, self.pwd)

    def db_connect(self):
        conn_str = self.get_conn_str()
        self.conn = pyodbc.connect(conn_str, autocommit=True)

    def db_close(self):
        self.conn.close()

    def db_execute(self, sql):
        self.db_connect()
        cursor = self.conn.cursor()
        cursor.execute(sql)
        self.db_close()

    def db_read(self, sql, chunksize=None):
        self.db_connect()
        results = read_sql(sql, self.conn, chunksize=chunksize)
        self.db_close()
        return results

    def db_read_pandas(self, sql, chunksize):
        self.db_connect()
        chunks =pd.read_sql_query(sql, self.conn, chunksize=chunksize)


    def db_insert(self, df, table_name, fast_executemany=False):
        self.db_connect()
        cursor = self.conn.cursor()
        column_name = [row.column_name for row in cursor.columns(table=table_name)]
        sql_str = 'insert into {0} ({1}) values ({2})'.format(table_name, ",".join(column_name), ",".join("?"*len(column_name)))
        cursor.close()
        del cursor

        iters, _ = divmod(df.shape[0], 1000)

        for i in range(iters+1):
            cursor = self.conn.cursor()
            cursor.fast_executemany = fast_executemany
            cursor.executemany(sql_str, df[i*1000:(i+1)*1000].values.tolist())
            cursor.close()
            del cursor

        self.db_close()


# ### insert examples ###
# import pyodbc
# import pandas as pd
# table_name= 'pytest'
# conn_str = 'DRIVER={SQL Server};SERVER=sldb;DATABASE=qdb;UID=solution;PWD=obc'
# conn = pyodbc.connect(conn_str, autocommit=True)
# cursor = conn.cursor()
# column_name = [row.column_name for row in cursor.columns(table=table_name)]
# sql_str = 'insert into {0} ({1}) values ({2})'.format(table_name, ",".join(column_name), ",".join("?"*len(column_name)))
#
# aa = pd.DataFrame({'cola':[1,3,5,4],'colb':['sef','23rd','3fss','3fsf'],'colc':[1.4,2.3,6.4,11.2]})
# cursor.executemany(sql_str, aa.as_matrix().tolist())
# cursor.close()
# del cursor
# conn.close()
