from qdata.dbmanager import SqlManager
from qinv import settings


class Asset:
    def __init__(self, asset_cls):
        self.asset_cls = asset_cls
        self.item_dict = settings.item_dict.get(self.asset_cls)
        self.is_initialize = False

    def initialize(self):
        raise NotImplementedError

    def get_asset_cls(self):
        return self.asset_cls

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

        print("[Asset] No Item in the DB list. Please Check [settings->item_dict] or [Item class]")
        return None

    def __setitem__(self, key, value):
        pass


class Equity(Asset):
    def __init__(self):
        super().__init__('equity')

    def initialize(self):
        sqlm = SqlManager()
        if self.item_dict['financial'] is None:
            sql_init_equity = 'select factor_nm from qinv..equitypitfinfactormstr'
            df_temp = sqlm.db_read(sql_init_equity)
            self.item_dict['financial'] = [i.lower() for i in df_temp.factor_nm]

        if self.item_dict['fc'] is None:
            pass

        self.is_initialize = True

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


class Commodity(Asset):
    def __init__(self):
        super().__init__('commodity')

    def initialize(self):
        pass

class Fx(Asset):
    def __init__(self):
        super().__init__('fx')

    def initialize(self):
        pass


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