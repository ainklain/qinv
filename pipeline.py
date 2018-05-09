import socket
import settings
from dbmanager import SqlManager
from datasource import Helper

# pipe = Pipeline()
# factor_obj = Factor() ?!
# pipe.add_pipeline('pipe_equity', universe=univ.univ_dict['equity'], item=item_obj)
class PipeInfo:
    """
        Pipeline을 통해 DB의 데이터를 가져오기 위한 필요 데이터관련 정보
        universe, characteristics, ...
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def addattr(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return_str = ""
        for key in self.__dict__.keys():
            return_str += "{}  ".format(key)
        return_str += '\n'

        return return_str


class PipeIO:
    def __init__(self):
        self.pipeline = dict()

    @staticmethod
    def get_table_id(name):
        my_ip = socket.gethostbyname(socket.gethostname())
        if my_ip in settings.ip_class.keys():
            table_id = name + '_' + settings.ip_class[my_ip]
        else:
            table_id = name + '_' + 'unknown'
        return table_id

    @staticmethod
    def is_stored(name):
        sqlm = SqlManager()
        sqlm.set_db_name('qpipe')

        table_id = PipeIO.get_table_id(name)
        df = sqlm.db_read("SELECT code_str FROM qpipe..Pipe_Information where table_id = '{}'".format(table_id))
        if df.empty:
            return False
        else:
            return True

    def load_pipeline(self, name, overwrite=False):
        """DB에 pipeline이 저장되어 있으면 True 아니면 False"""
        sqlm = SqlManager()
        sqlm.set_db_name('qpipe')

        table_id = PipeIO.get_table_id(name)
        df = sqlm.db_read("SELECT code_str FROM qpipe..Pipe_Information where table_id = '{}'".format(table_id))

        if df.empty:
            print("[load_pipeline] Not exists pipe:'{}'".format(name))
            return False
        else:
            code_dict = eval(df.loc[0][0].replace("?", "'"))
            decoded_dict = Helper.bulk_decode(**code_dict)

            if (overwrite is True) or (name not in self.pipeline.keys()):
                self.pipeline[name] = PipeInfo(table_id=table_id, **decoded_dict)
                print("[load_pipeline] Successfully loaded")
            else:
                print("[load_pipeline] Already exists or Overwrite is not allowed. Use 'overwrite=True' if you want.")

            return True

    def store_pipeline(self, name, sch_obj):
        sqlm = SqlManager()
        sqlm.set_db_name('qpipe')

        pipe_dict = self.pipeline[name]
        enc_dict = Helper.bulk_encode(
            universe=pipe_dict.universe,
            item=pipe_dict.item,
            schedule=sch_obj)
        enc_dict = str(enc_dict).replace("'", "?")

        sch_str = "{}/{}/{}/'{}'".format(sch_obj.begin_d, sch_obj.end_d, sch_obj.type_, sch_obj.desc_).replace("'", "")

        sqlm.db_execute("""        
        DELETE FROM QPIPE..PIPE_INFORMATION WHERE TABLE_ID = '{table_id}' AND SCHEDULE_STR = '{sch}'
        INSERT INTO qpipe..pipe_information VALUES ('{table_id}','{sch}','{code_str}', getdate())

        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME=N'{table_id}')
        BEGIN
        create table {table_id} (item_nm varchar(40), infocode int, eval_d date, value_ float
        primary key(item_nm, infocode, eval_d)
        )
        END
        ELSE BEGIN
        truncate table {table_id}
        END
        """.format(table_id=pipe_dict.table_id, code_str=enc_dict, sch=sch_str))

        univ_str = '&'.join(pipe_dict.universe)
        schedule_str = pipe_dict.code.replace("'", "''")

        for item_key in pipe_dict.item.keys():
            item_str = '&'.join(pipe_dict.item[item_key].item_set)
            item_expr = pipe_dict.item[item_key].expr

            sql_code = sqlm.db_read("select qinv.dbo.FS_CodeAssembler('{}', '{}', '{}', '{}')".format(
                univ_str, item_str, item_expr, schedule_str))
            sql_code = sql_code.values[0][0]

            # return_dict[item_key] = sqlm.db_execute(sql_code)

            pipe_dict.addattr(sql_code=sql_code)
            sqlm.db_execute(
                """insert into {} \n{}""".format(pipe_dict.table_id, sql_code.replace('[item_nm]', item_key)))

    def get_item(self, name, item_id):
        sqlm = SqlManager()
        sqlm.set_db_name('qpipe')

        table_id = self.pipeline[name].table_id
        if item_id not in self.pipeline[name].item.keys():
            print('No such item exists')
            return None

        item_data = sqlm.db_read("select * from {} where item_nm='{}'".format(table_id, item_id))
        if item_data.empty:
            print('You should [run_pipeline] first.')
            return None
        return item_data

    def get_code(self, name, schedule):
        """어디 쓰이지??"""
        sqlm = SqlManager()
        sqlm.set_db_name('qinv')
        mypipe = self.pipeline[name]
        univ_str = '&'.join(mypipe.universe)
        schedule_str = schedule.code.replace("'", "''")

        code_list = dict()
        for item_key in mypipe.item.keys():
            item_str = '&'.join(mypipe.item[item_key].item_set)
            item_expr = mypipe.item[item_key].expr

            sql_code = sqlm.db_read("select qinv.dbo.FS_CodeAssembler('{}', '{}', '{}', '{}')".format(
                univ_str, item_str, item_expr, schedule_str))
            sql_code = sql_code.values[0][0]
            code_list[item_key] = sql_code
        return code_list

    def __getitem__(self, name):
        return self.pipeline[name]

    def __repr__(self):
        return_str = "[[[ pipe name : attributes ]]]\n"
        for name in self.pipeline.keys():
            return_str += "{} : ".format(name)
            return_str += self.pipeline[name].__repr__()

        return return_str


class Pipeline(PipeIO):
    """
        받고자 하는 데이터를 abstract 수준으로 받아서 pipeline별로 add_pipeline을 통해 추가
        run_pipeline을 통해 실제로 DB연결을 통해 데이터를 한꺼번에 받아오기

        pipeline은 dictionary로 관리되며 key는 pipeline의 이름, value는 DataInfo타입의 class object로 저장   

        pipeline class object 자체는 하나만 생성하되, 목적별로 다른 데이터를 pipeline name으로 구별
        run_pipeline도 name별로 그떄그때 가져오는 방식
    """
    def __init__(self, name=None):
        self.pipeline = dict()
        if name is not None:
            self.load_pipeline(name=name)

    def add_pipeline(self, name, **kwargs):
        self.pipeline[name] = PipeInfo(**kwargs)
        self.pipeline[name].addattr(table_id=PipeIO.get_table_id(name))

    def run_pipeline(self, name, schedule=None, store_db=False):
        #
        is_loaded = self.load_pipeline(name, overwrite=False)
        if (store_db is True) or (is_loaded is False):
            if schedule is None:
                print('Schedule should be set.')
                return None
            self.store_pipeline(name=name, sch_obj=schedule)
