

# from qdata.io import IO



class Universe_tbd:
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

# # example
# def run_test():
#     equity_obj = Equity()
#     equity_obj.initialize()
#     mktcap = equity_obj['mktcap']
#     be = equity_obj['be']
#     bm = be / mktcap
#
#     univ = Universe()
#     univ.add_univ('equity', 'us_all')
#     # univ.add_univ('equity', 'kr_sample')
#     # univ.add_univ('equity', 'us_sample')
#     # univ.add_univ('equity', 'us_all')
#
#     pipe = Pipeline()
#     pipe.add_pipeline('pipe_equity', universe=univ['equity'], item={'bm':bm, 'be':be})
#     pipe.add_pipeline('pipe_equity2', universe=univ['equity'], item={'mktcap': mktcap})
#
#     # sch_obj = Schedule('2017-01-01', '2017-03-01', type_='spec', days=[2, 6, 7])
#     sch_obj = Schedule('2015-01-01', '2018-01-01', type_='end', freq_='m')
#
#     begin_t = time.time()
#     pipe.run_pipeline('pipe_equity2', schedule=sch_obj)
#     end_t = time.time()
#
#     data_be = pipe.get_item('pipe_equity', 'be')
#     data_bm = pipe.get_item('pipe_equity', 'bm')
#     data_mktcap = pipe.get_item('pipe_equity2', 'mktcap')
#     return
#