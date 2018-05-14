import settings
import socket
from dbmanager import SqlManager


__version__ = '1.1.0'


class IO:
    @staticmethod
    def get_table_id(name):
        my_ip = socket.gethostbyname(socket.gethostname())
        if my_ip in settings.ip_class.keys():
            table_id = name + '_' + settings.ip_class[my_ip]
        else:
            table_id = name + '_' + 'unknown'
        return table_id


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



class ItemDefault:
    def __init__(self, item_cls=None, item_nm=None):
        self.item_set = set()
        self.expr = ''
        self.asset_cls = ''
        if item_cls is not None:
            self.item_set.add('{}::{}'.format(item_cls, item_nm))
            self.expr = '[{}]'.format(item_nm)

    def __add__(self, other):
        return_item = Item()
        assert self.asset_cls == other.asset_cls
        return_item.asset_cls = self.asset_cls
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '+' + other.expr + ')'

        return return_item

    def __sub__(self, other):
        return_item = Item()
        assert self.asset_cls == other.asset_cls
        return_item.asset_cls = self.asset_cls
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '-' + other.expr + ')'

        return return_item

    def __mul__(self, other):
        return_item = Item()
        assert self.asset_cls == other.asset_cls
        return_item.asset_cls = self.asset_cls
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '*' + other.expr + ')'

        return return_item

    def __truediv__(self, other):
        return_item = Item()
        assert self.asset_cls == other.asset_cls
        return_item.asset_cls = self.asset_cls
        return_item.item_set = self.item_set | other.item_set
        return_item.expr = '(' + self.expr + '/ nullif(' + other.expr + ',0))'

        return return_item

    def __repr__(self):
        return_str = 'item string: {} \nexpression: {}\n'.format(self.item_set, self.expr)
        return return_str


class Item(ItemDefault):
    """
        DB에서 ITem을 가져오기 위한 최소한의 구분 정보를 담은 class
    """
    def __init__(self, item_cls=None, item_nm=None, **kwargs):
        super().__init__(item_cls, item_nm)
        self.asset_cls = kwargs.get('asset_cls')
        self._set_attr(**kwargs)

    def _set_attr(self, **kwargs):
        item_set = kwargs.get('item_set')
        expr = kwargs.get('expr')
        if item_set is not None and expr is not None:
            self.item_set = item_set
            self.expr = expr


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
            self.desc_ = "{{'freq_': '{}'}}".format(kwargs['freq_'])
        elif self.type_ == 'spec':
            assert 'days' in key_list
            if 'months' in key_list:
                self.code = self._gen_code(days=kwargs['days'], months=kwargs['months'])
                self.desc_ = "{{'days': {}, 'months': {} }}".format(kwargs['days'], kwargs['months'])
            else:
                self.code = self._gen_code(days=kwargs['days'])
                self.desc_ = "{{'days': {}}}".format(kwargs['days'])
        elif self.type_ == 'interval':
            assert 'freq_' in key_list
            assert 'interval' in key_list
            self.code = self._gen_code(freq_=kwargs['freq_'], interval=kwargs['interval'])
            self.desc_ = "{{'freq_': '{}', 'interval': {}}}".format(kwargs['freq_'], kwargs['interval'])
        else:
            pass


class Asset:
    def __init__(self):
        self.asset_cls = self.__class__.__name__.lower()

class Equity(Asset):
    """
        equity 관련 데이터 처리를 위한 클래스
        하나의 equity 객체를 통해 여러 데이터를 가져올 수 있게끔
    """
    def __init__(self):
        super().__init__()
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

                return Item(key, item_nm, asset_cls=self.asset_cls)

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



# univ = Universe()
# univ.add_univ('equity', 'us_all')
# univ.add_univ('equity', 'kr_all')
class Universe:
    """
        투자(전략만들기)하려는 유니버스를 관리해주는 클래스
        add_univ를 이용하여 DB에 저장되어 있는 universe 명을 추가
        투자 asset class별로 따로 관리하며 같은 asset에 대해서는 리스트로 관리
    """

    def __init__(self, **kwargs):
        self.univ_dict = dict()
        for key in kwargs.keys():
            if isinstance(kwargs[key], str):
                self.add_univ(key, kwargs[key])
            elif isinstance(kwargs[key], list):
                for item in kwargs[key]:
                    self.add_univ(key, item)

    def add_univ(self, type_, name):
        """
        :param type_: asset class명 (i.e. equity, commodity) 
        :param name: db에 저장되어 있는 universe 명
        :return: 없음

        asset class와 univ_name이 사전에 DB에 저장되어 있어야 하는지를 assert를 통해 확인
         이 부분에서 에러가 발생할 경우 settings에서 DB에 있는 내용을 추가해주거나 혹은 오타 확인
         settings의 값을 자동 추가 혹은 변경하는 방법은 추후 initialize()등을 만들어서 할 수도 있음
        """
        if (type_ not in settings.univ_dict_sql.keys()) or (name not in settings.univ_dict_sql[type_]):
            print("""Asset:{}, Univ:{} cannot be added. Please check settings.univ_dict_sql. 
            """.format(type_, name))
            return None

        if type_ not in self.univ_dict.keys():
            self.univ_dict[type_] = [name]
        else:
            self.univ_dict[type_].append(name)

        return True

    def __getitem__(self, item):
        if item in self.univ_dict.keys():
            return self.univ_dict[item]

    def __repr__(self):
        return_str = "++universe dictionary++\n"
        for key in self.univ_dict.keys():
            return_str += "asset class: {}\n  - univ_nm: {}\n\n".format(key, ", ".join(self.univ_dict[key]))

        return return_str

