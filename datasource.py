import settings
import socket
from dbmanager import SqlManager

__version__ = '1.1.0'

class Item():
    """
        DB에서 ITem을 가져오기 위한 최소한의 구분 정보를 담은 class
    """
    def __init__(self, item_set=None, expr=None, **kwargs):
        self.item_set = set()
        self.expr = ''


        else item_set is None or expr is None:
            print(kwargs)
            self._set_attr(**kwargs)

    def _set_attr(self, **kw):
        assert 'class_nm' in kw.keys()
        assert 'item_nm' in kw.keys()
        self.item_set.add('{}::{}'.format(kw['class_nm'], kw['item_nm']))
        self.expr = '[{}]'.format(kw['item_nm'])

    def __add__(self, other):
        return_item = Item()
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '+' + other.expr + ')'

        return return_item

    def __sub__(self, other):
        return_item = Item()
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '-' + other.expr + ')'

        return return_item

    def __mul__(self, other):
        return_item = Item()
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '*' + other.expr + ')'

        return return_item

    def __truediv__(self, other):
        return_item = Item()
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '/ nullif(' + other.expr + ',0))'

        return return_item

    def __repr__(self):
        return_str = 'item string: {} \nexpression: {}\n'.format(self.item_set, self.expr)
        return return_str


class Item_old:
    """
        DB에서 ITem을 가져오기 위한 최소한의 구분 정보를 담은 class
    """
    def __init__(self, class_nm=None, item_nm=None):
        self.item_set = set()
        self.expr = ''
        if class_nm is not None:
            self.item_set.add('{}::{}'.format(class_nm, item_nm))
            self.expr = '[{}]'.format(item_nm)

    def __add__(self, other):
        return_item = Item()
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '+' + other.expr + ')'

        return return_item

    def __sub__(self, other):
        return_item = Item()
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '-' + other.expr + ')'

        return return_item

    def __mul__(self, other):
        return_item = Item()
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '*' + other.expr + ')'

        return return_item

    def __truediv__(self, other):
        return_item = Item()
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '/ nullif(' + other.expr + ',0))'

        return return_item

    def __repr__(self):
        return_str = 'item string: {} \nexpression: {}\n'.format(self.item_set, self.expr)
        return return_str


# sch = Schedule('2017-01-01','2017-03-01',type_='end',freq_='m')
# sch = Schedule('2017-01-01','2017-03-01',type_='spec',days=[2, 6, 7])
class Schedule:
    """
        가져오려는 데이터의 기간/주기 설정 클래스
        필수파라미터
        begin_d: 시작날짜 ('yyyy-mm-dd' 형식)
        end_d: 끝날짜 ('yyyy-mm-dd' 형식)
        type_: 어떤 형식으로 받아올지 결정 (end | start | spec | interval)
        if type_ in (end | start) : freq_ ('y','q','m','w','d')
                    (spec)        : days ([2, 3, ...]   months if needed)
                    (interval)    : freq_, interval (integer)        
    """
    def __init__(self, begin_d, end_d, type_=None, **kwargs):
        self.code = None
        self.desc_ = None
        self.begin_d = begin_d
        self.end_d = end_d
        self.type_ = type_
        self.set_schedule(**kwargs)

    def _attr_set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._gen_code()

    def _gen_code(self, freq_=None, interval=None, days=None, months=None):
        sqlm = SqlManager()
        sqlm.set_db_name('qinv')

        freq_str = 'Null'
        days_str = 'Null'
        months_str = 'Null'
        interval_str = 'Null'

        if freq_ is not None:
            freq_str = "'" + freq_ + "'"

        if interval is not None:
            interval_str = "'" + str(interval) + "'"

        if days is not None:
            days_str = "'" + ','.join([str(i) for i in days]) + "'"

        if months is not None:
            months_str = "'" + ','.join([str(i) for i in months]) + "'"

        sql_ = "select qinv.dbo.FS_ScheduleCodeMaker('"+self.begin_d+"', '" + self.end_d + "', '" + self.type_ + \
               "', " + freq_str + ", " + interval_str + ", " + days_str + ", " + months_str + ")"
        code = sqlm.db_read(sql_)
        code = code.values[0][0]

        return code

    def set_schedule(self, **kwargs):
        """
        :param kwargs:
            가져오려는 데이터의 기간/주기 설정 클래스
            필수파라미터
            begin_d: 시작날짜 ('yyyy-mm-dd' 형식)
            end_d: 끝날짜 ('yyyy-mm-dd' 형식)
            type_: 어떤 형식으로 받아올지 결정 (end | start | spec | interval)
            if type_ in (end | start) : freq_ ('y','q','m','w','d')
                    (spec)        : days ([2, 3, ...]   months if needed)
                        (interval)    : freq_, interval (integer)
         
        """
        key_list = [i.lower() for i in kwargs.keys()]
        if 'begin_d' in key_list:
            self.begin_d = kwargs['begin_d']
        if 'end_d' in key_list:
            self.end_d = kwargs['end_d']
        if 'type_' in key_list:
            self.type_ = kwargs['type_']

        if self.type_ in ('end', 'start'):
            assert 'freq_' in key_list
            self.code = self._gen_code(freq_=kwargs['freq_'])
            self.desc_ = 'freq_: {}'.format(kwargs['freq_'])
        elif self.type_ == 'spec':
            assert 'days' in key_list
            if 'months' in key_list:
                self.code = self._gen_code(days=kwargs['days'], months=kwargs['months'])
                self.desc_ = 'days: {} // months: {}'.format(kwargs['days'], kwargs['months'])
            else:
                self.code = self._gen_code(days=kwargs['days'])
                self.desc_ = 'days: {}'.format(kwargs['days'])
        elif self.type_ == 'interval':
            assert 'freq_' in key_list
            assert 'interval' in key_list
            self.code = self._gen_code(freq_=kwargs['freq_'], interval=kwargs['interval'])
            self.desc_ = 'freq_: {} // interval: {}'.format(kwargs['freq_'], kwargs['interval'])
        else:
            pass


class Equity:
    """
        equity 관련 데이터 처리를 위한 클래스
        하나의 equity 객체를 통해 여러 데이터를 가져올 수 있게끔
    """
    def __init__(self):
        self.item_dict = settings.item_dict['equity']
        self.is_initialize = False

    def initialize(self):
        sqlm = SqlManager()
        if self.item_dict['financial'] is None:
            sql_init_equity = 'select factor_nm from qinv..equitypitfinfactormstr'
            df_temp = sqlm.db_read(sql_init_equity)
            self.item_dict['financial'] = [i.lower() for i in df_temp.factor_nm]

        if self.item_dict['fc'] is None:
            pass

        self.is_initialize = True

    def __getitem__(self, item_nm):
        if not self.is_initialize:
            print('It should be initialize. Use initialize().')
            return None

        for key in self.item_dict.keys():
            if self.item_dict[key] is None:
                continue

            if item_nm in self.item_dict[key]:
                print('{} {}'.format(key, item_nm))
                return Item(key, item_nm)

        print("[Equity] No Item in the DB list. Please Check [settings->item_dict] or [Item class]")
        return None

    def __setitem__(self, key, value):
        pass

    def __repr__(self):
        return_str = "[Equity] Initialized: {}\n".format(str(self.is_initialize))
        if self.is_initialize:
            for key in self.item_dict.keys():
                return_str += "<< {} >>\n".format(key)

                if self.item_dict[key] is None:
                    continue

                for i, item in enumerate(self.item_dict[key]):
                    if i > 0:
                        return_str += " / "
                        if i % 10 == 0:
                            return_str += "\n"

                    return_str += "{0:^8}".format(item)

                return_str += "\n\n"
        return return_str


class DataInfo:
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


# pipe = Pipeline()
# factor_obj = Factor() ?!
# pipe.add_pipeline('pipe_equity', universe=univ.univ_dict['equity'], item=item_obj)
class Pipeline:
    """
        받고자 하는 데이터를 abstract 수준으로 받아서 pipeline별로 add_pipeline을 통해 추가
        run_pipeline을 통해 실제로 DB연결을 통해 데이터를 한꺼번에 받아오기

        pipeline은 dictionary로 관리되며 key는 pipeline의 이름, value는 DataInfo타입의 class object로 저장   

        pipeline class object 자체는 하나만 생성하되, 목적별로 다른 데이터를 pipeline name으로 구별
        run_pipeline도 name별로 그떄그때 가져오는 방식
    """

    def __init__(self):
        self.pipeline = dict()

    def add_pipeline(self, name, **kwargs):
        self.pipeline[name] = DataInfo(**kwargs)
        my_ip = socket.gethostbyname(socket.gethostname())
        if my_ip in settings.ip_class.keys():
            table_id = name + '_' + settings.ip_class[my_ip]
        else:
            table_id = name + '_' + 'unknown'
        self.pipeline[name].addattr(table_id=table_id)

    def run_pipeline(self, name, schedule):
        sqlm = SqlManager()
        sqlm.set_db_name('qpipe')

        mypipe = self.pipeline[name]
        assert hasattr(mypipe, 'universe')
        assert hasattr(mypipe, 'item')


        sqlm.db_execute("""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME=N'{table_id}')
        BEGIN
        create table {table_id} (item_nm varchar(40), infocode int, eval_d date, value_ float
        primary key(item_nm, infocode, eval_d)
        )
        END
        ELSE BEGIN
        truncate table {table_id}
        END
        """.format(table_id=mypipe.table_id))


        univ_str = '&'.join(mypipe.universe)
        schedule_str = schedule.code.replace("'", "''")

        for item_key in mypipe.item.keys():
            item_str = '&'.join(mypipe.item[item_key].item_set)
            item_expr = mypipe.item[item_key].expr

            sql_code = sqlm.db_read("select qinv.dbo.FS_CodeAssembler('{}', '{}', '{}', '{}')".format(
                univ_str, item_str, item_expr, schedule_str))
            sql_code = sql_code.values[0][0]

            # return_dict[item_key] = sqlm.db_execute(sql_code)

            mypipe.addattr(sql_code=sql_code)
            sqlm.db_execute("""insert into {} \n{}""".format(mypipe.table_id, sql_code.replace('[item_nm]', item_key)))

    def get_code(self, name, schedule):
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

    def get_item(self, name, item_id):
        sqlm = SqlManager()
        sqlm.set_db_name('qpipe')

        table_id = self.pipeline[name].table_id
        if item_id not in self.pipeline[name].item.keys():
            print('No such item exists')
            return None

        item_data = sqlm.db_read("select * from {} where item_nm='{}'".format(table_id, item_id))
        return item_data

    def __getitem__(self, name):
        return self.pipeline[name]

    def __repr__(self):
        return_str = "[[[ pipe name : attributes ]]]\n"
        for name in self.pipeline.keys():
            return_str += "{} : ".format(name)
            return_str += self.pipeline[name].__repr__()

        return return_str


# univ = Universe()
# univ.add_univ('equity', 'us_all')
# univ.add_univ('equity', 'kr_all')
class Universe:
    """
        투자(전략만들기)하려는 유니버스를 관리해주는 클래스
        add_univ를 이용하여 DB에 저장되어 있는 universe 명을 추가
        투자 asset class별로 따로 관리하며 같은 asset에 대해서는 리스트로 관리
    """

    def __init__(self):
        self.univ_dict = dict()

    def add_univ(self, type_, name):
        """
        :param type_: asset class명 (i.e. equity, commodity) 
        :param name: db에 저장되어 있는 universe 명
        :return: 없음

        asset class와 univ_name이 사전에 DB에 저장되어 있어야 하는지를 assert를 통해 확인
         이 부분에서 에러가 발생할 경우 settings에서 DB에 있는 내용을 추가해주거나 혹은 오타 확인
         settings의 값을 자동 추가 혹은 변경하는 방법은 추후 initialize()등을 만들어서 할 수도 있음
        """
        assert type_ in settings.univ_dict_sql.keys()
        assert name in settings.univ_dict_sql[type_]

        if type_ not in self.univ_dict.keys():
            self.univ_dict[type_] = [name]
        else:
            self.univ_dict[type_].append(name)

    def __getitem__(self, item):
        if item in self.univ_dict.keys():
            return self.univ_dict[item]

    def __repr__(self):
        return_str = "++universe dictionary++\n"
        for key in self.univ_dict.keys():
            return_str += "asset class: {}\n  - univ_nm: {}\n\n".format(key, ", ".join(self.univ_dict[key]))

        return return_str





# def get_data_from_tablename(source_name, module_name=None, **kwargs):
#     if module_name is None:
#         module_name = datasource
#
#     try:
#         return getattr(module_name, source_name)(**kwargs)
#     except ImportError:
#         print('GBI MODULE ERROR: You have not imported the {} module.'.format(module_name))
#         exit()


# class SuriData:
#     def __init__(self, information=None, manual=False):
#         if information is None:
#             print('SURIDATA: No information to initialize.')
#             self.univ_cd = None
#             self.stock_df = None
#             self.suri_search = None
#             self.suri_wgt = None
#             self.bond_df = None
#             self.discount_rate = None
#             self.spot_curve = None
#             self.goal_beta_matrix = None
#             self.proj_dir = None
#             self.rates = None
#         else:
#             if manual is False:
#                 self.set_suri_data(information)
#             else:
#                 self.set_suri_data_by_manual(information)
#
#     def set_suri_data(self, information):
#         self.proj_dir = information['proj_dir']
#         self.univ_cd = information['univ_cd']
#         self.stock_df = get_data_from_tablename(information['asset_data'], stock_list=information['stock_list'])
#         self.suri_search, self.suri_wgt = get_data_from_tablename(information['suri_data'],
#                                                                   univ_cd=information['univ_cd'])
#         self.spot_curve = read_txt(information['spot_curve'], information['proj_dir'])
#         self.bond_df, self.discount_rate = get_data_from_tablename(information['bond_model'],
#                                                                    module_name=sys.modules[__name__]
#                                                                    , spot_df=self.spot_curve,
#                                                                    bond_info=information['bond_info'],
#                                                                    year=information['year'])
#         self.goal_beta_matrix = get_data_from_tablename(information['goal_beta_matrix'],
#                                                         module_name=sys.modules[__name__]
#                                                         , spot_df=self.spot_curve, year=information['year'],
#                                                         infl_r=information['infl_r'])
#         self.rates = get_data_from_tablename(information['rates'])
#
#     def set_suri_data_by_manual(self):
#         print('[SuriData][set_suri_data_by_manual] 여기에 DIY 코딩하시면 됩니다.')
#
#     def __repr__(self):
#         return_str = ""
#         return_str += "UNIV_CD:{}\n".format(self.univ_cd)
#         return_str += "Data List: " + ", ".join(self.__dict__.keys())
#         return_str += "\n"
#         return_str += "stock_df:\n{}\n".format(self.stock_df.head(3))
#         return_str += "bond_df:\n{}\n".format(self.bond_df.head(3))
#         return_str += "suri_search:\n{}\n".format(self.suri_search.head(3))
#         return_str += "suri_wgt:\n{}\n".format(self.suri_wgt.head(3))
#         return return_str

