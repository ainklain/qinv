from qdata.dbmanager import SqlManager


def obj_to_dbformat(obj, default='Null', sep=",", end="'"):
    if isinstance(obj, list):
        return_str = end + sep.join([str(i) for i in obj]) + end
    elif obj is None:
        return_str = default
    else:
        return_str = end + str(obj) + end

    return return_str


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
        sql_ = "select qinv.dbo.FS_ScheduleCodeMaker({}, {}, {}, {}, {}, {}, {})".format(
            *tuple(map(obj_to_dbformat, [self.begin_d, self.end_d, self.type_, freq_, interval, days, months])))

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

