import time

# from qdata.pipeline import PipeInfo
from qinv import settings
import socket
from qdata.dbmanager import SqlManager
from qinv.asset import Item
from qinv.schedule import Schedule
from qinv.universe import Universe

__version__ = '1.1.0'


def get_table_id(name, table_owner=None):
    if table_owner is not None:
        table_id = name + '_' + table_owner
    else:
        my_ip = socket.gethostbyname(socket.gethostname())
        table_id = name + '_' + settings.ip_class.get(my_ip, 'unknown')

    return table_id


class IO:
    def __init__(self):
        self.sqlm = SqlManager()
        self.sqlm.set_db_name('qpipe')

    def create_table(self, table_id, table_structure):
        # table_structure: {'columns': list of tuples for columns (col_name, col_type), 'pk':list of pk}
        table_struct_code = ""
        for (col_nm, col_type) in table_structure['columns']:
            table_struct_code += col_nm + " " + col_type + ",\n"
        if 'pk' in table_structure.keys():
            table_struct_code += "primary key(" + ", ".join(table_structure['pk']) + ")"

        self.sqlm.db_execute("""        
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME=N'{table_id}')
        BEGIN
        create table {table_id} ({table_structure})
        END
        ELSE BEGIN
        truncate table {table_id}
        END
        """.format(table_id=table_id, table_structure=table_struct_code))

    def load(self, *args, **kwargs):
        raise NotImplementedError

    def store(self, *args, **kwargs):
        raise NotImplementedError


class TestingIO(IO):
    def __init__(self):
        super().__init__()

    def store(self, order_table, process_nm='backtest'):
        import time
        sqlm = self.sqlm

        table_id = get_table_id(process_nm)
        table_structure = {'columns': [('eval_d', 'date'),
                                       ('infocode', 'int'),
                                       ('wgt', 'float')],
                           'pk': ['eval_d', 'infocode']}

        self.create_table(table_id, table_structure)
        print('Table {} created.'.format(table_id))

        s = time.time()
        sqlm.db_insert(order_table, table_id, fast_executemany=True)
        print('Inserted.')
        e = time.time()
        print("calculation time: {0:.2f} sec".format(e - s))

    def execute(self, process_nm='backtest'):
        if process_nm in ['backtest']:
            table_id = get_table_id(process_nm)
            sql_ = """            
            select a.eval_d, 100 + sum(cum_w) as w, 100 + sum(cum_y) as y,  (100 + sum(cum_y)) / (100 + sum(cum_w)) - 1 as y
            from (
            select  a.infocode, a.eval_d as calc_d, e.eval_d, y
                , 100 * a.wgt * exp(sum(log(1+isnull(y, 0.))) 
                    over (	partition by a.infocode, a.eval_d 
                            order by e.eval_d rows 
                            between unbounded preceding and current row)) as cum_y
                , 100 * a.wgt * isnull(exp(sum(log(1+isnull(y, 0.))) 
                    over (	partition by a.infocode, a.eval_d 
                            order by e.eval_d rows 
                            between unbounded preceding and 1 preceding)), 1) as cum_w
                from (
                    select eval_d, isnull(lead(eval_d, 1) over (order by eval_d), dateadd(day, 31, eval_d)) as fwd_d
                        from qpipe..{table_id}		
                        group by eval_d
                ) D
                join (select * from qpipe..{table_id} A) a
                on D.eval_d = A.eval_d	
                join qinv..EquityTradeDate E
                on a.infocode = E.Infocode and E.Eval_d > d.eval_d and E.Eval_d <= d.fwd_d
                left join qinv..EquityReturnDaily R
                on a.infocode = r.infocode and e.Eval_d = r.marketdate 
            ) A
            group by a.eval_d
            order by a.eval_d
            """.format(table_id=table_id)

            df = self.sqlm.db_read(sql_)
            return df


# pipe = Pipeline()
# factor_obj = Factor() ?!
# pipe.add('pipe_equity', universe=univ.univ_dict['equity'], item=item_obj)
class PipeInfo:
    """
        Pipeline을 통해 DB의 데이터를 가져오기 위한 필요 데이터관련 정보
        universe, characteristics, ...
    """
    def __init__(self, **kwargs):
        self.add_attr(**kwargs)

    def add_attr(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return_str = ""
        for key in self.__dict__.keys():
            return_str += "{}  ".format(key)
        return_str += '\n'

        return return_str


class PipeIO(IO):
    def __init__(self):
        super().__init__()
        self.pipeline = dict()
        self._initialize_info()

    def _initialize_info(self):
        self.pipe_info = self.sqlm.db_read("select table_id, schedule_str, code_str from qpipe..pipe_information")

    def is_stored(self, table_id):
        return table_id in list(self.pipe_info['table_id'])

    def load(self, name, table_owner=None):
        """DB에 pipeline이 저장되어 있으면 True 아니면 False"""

        table_id = get_table_id(name, table_owner)

        if not self.is_stored(table_id):
            print("[PipeIO][load] Not exists pipe:'{}' for table_owner:'{}'".format(name, table_owner))
            return False
        else:
            stored = self.pipe_info.loc[self.pipe_info['table_id'] == table_id, 'code_str']
            code_dict = eval(stored.values[0].replace("?", "'"))
            decoded_dict = Helper.bulk_decode(**code_dict)

            self.pipeline[name] = PipeInfo(table_id=table_id, added_item=dict(), **decoded_dict)
            print("[PipeIO][load] Successfully loaded")
            return True

    def _insert_item_to_db(self, pipe_dict, sch_obj, chunksize, update=False):
        if update is True:
            items = pipe_dict.added_item
        else:
            items = pipe_dict.item

        # equities/commodities/fx 전략 구분하여 적용 위함
        for asset_cls in pipe_dict.universe.univ_dict.keys():
            univ_str = '&'.join(pipe_dict.universe[asset_cls])
            schedule_str = sch_obj.code.replace("'", "''")

            for item_key in items.keys():
                if items[item_key].asset_cls != asset_cls:
                    continue

                item_str = '&'.join(items[item_key].item_set)
                item_expr = items[item_key].expr
                sql_code = """exec qinv.dbo.SP_Run_EquityItems '{}', '{}', '{}', '{}', '{}', '{}', {}""".format(
                    univ_str, item_str, item_expr, schedule_str, pipe_dict.table_id, item_key, chunksize)
                pipe_dict.add_attr(sql_code=sql_code)

                st = time.time()
                print(sql_code)
                self.sqlm.db_execute(sql_code)
                et = time.time()

                print("Successfully Stored. [{0:.2f} sec]".format(et-st))

    def store(self, name, sch_obj, chunksize=10000):
        # ENCODING
        pipe_dict = self.pipeline[name]
        enc_dict = Helper.bulk_encode(
            universe=pipe_dict.universe,
            item=pipe_dict.item,
            schedule=sch_obj)
        enc_dict = str(enc_dict).replace("'", "?")

        sch_str = "{}/{}/{}/'{}'".format(sch_obj.begin_d, sch_obj.end_d, sch_obj.type_, sch_obj.desc_).replace("'", "")

        # SAVE INFORMATION ABOUT PIPE
        self.sqlm.db_execute("""        
        DELETE FROM QPIPE..PIPE_INFORMATION WHERE TABLE_ID = '{table_id}'
        INSERT INTO qpipe..pipe_information VALUES ('{table_id}','{sch}','{code_str}', getdate())
        """.format(table_id=pipe_dict.table_id, code_str=enc_dict, sch=sch_str))

        self._initialize_info()

        # CREATE TABLE IF NOT EXISTS
        table_structure = {'columns': [('item_nm', 'varchar(40)'),
                                       ('infocode', 'int'),
                                       ('eval_d', 'date'),
                                       ('value_', 'float')],
                           'pk': ['item_nm', 'infocode', 'eval_d']}
        self.create_table(table_id=pipe_dict.table_id, table_structure=table_structure)

        self._insert_item_to_db(pipe_dict, sch_obj, chunksize)
        # equities/commodities/fx 전략 구분하여 적용 위함

        pipe_dict.add_attr(schedule=sch_obj)

    def get_item(self, name, **kwargs):
        item_id = kwargs.get('item_id', None)
        sqlm = SqlManager()
        sqlm.set_db_name('qpipe')

        table_id = self.pipeline[name].table_id
        if item_id is None:
            item_data = sqlm.db_read("select * from {}".format(table_id))
        else:
            if item_id not in self.pipeline[name].item.keys():
                print('No such item exists')
                return None

            item_data = sqlm.db_read("select * from {} where item_nm='{}'".format(table_id, item_id))

        if item_data.empty:
            print('You should [run] first.')
            return None
        return item_data

    def __getitem__(self, name):
        return self.pipeline[name]

    def __repr__(self):
        return_str = "[[[ pipe name : attributes ]]]\n"
        for name in self.pipeline.keys():
            return_str += "{} : ".format(name)
            return_str += self.pipeline[name].__repr__()

        return return_str


class AssetIO(IO):
    def __init__(self):
        super().__init__()


class Helper:
    @staticmethod
    def encode(obj):
        return {'data': obj.__dict__, 'cls': obj.__class__.__name__}

    @staticmethod
    def bulk_encode(**kwargs):
        encoded_dict = dict()
        for key in kwargs.keys():
            if isinstance(kwargs[key], dict):
                encoded_dict[key] = Helper.bulk_encode(**kwargs[key])
                continue
            encoded_dict[key] = Helper.encode(kwargs[key])
        return encoded_dict

    @staticmethod
    def decode(obj_str_dict):
        cls = obj_str_dict['cls']
        obj_dict = obj_str_dict['data']
        if cls.lower() in ['item']:
            obj = Item(**obj_dict)
        elif cls.lower() in ['univ', 'universe']:
            obj = Universe()
            obj.univ_dict = obj_dict['univ_dict']
        elif cls.lower() in ['sch', 'schedule']:
            obj = Schedule(begin_d=obj_dict['begin_d'],
                           end_d=obj_dict['end_d'],
                           type_=obj_dict['type_'],
                           **eval(obj_dict['desc_']))
        return obj

    @staticmethod
    def bulk_decode(**kwargs):
        decoded_dict = dict()
        for key in kwargs.keys():
            if 'data' not in kwargs[key].keys():
                decoded_dict[key] = Helper.bulk_decode(**kwargs[key])
                continue

            decoded_dict[key] = Helper.decode(kwargs[key])

        return decoded_dict







# sch = Schedule('2017-01-01','2017-03-01',type_='end',freq_='m')
# sch = Schedule('2017-01-01','2017-03-01',type_='spec',days=[2, 6, 7])


# univ = Universe()
# univ.add_univ('equity', 'us_all')
# univ.add_univ('equity', 'kr_all')

