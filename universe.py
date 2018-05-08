import time
from datasource import Equity, Schedule, Universe, Pipeline


# example
def run_test():
    equity_obj = Equity()
    equity_obj.initialize()
    mktcap = equity_obj['mktcap']
    be = equity_obj['be']
    bm = be / mktcap

    univ = Universe()
    univ.add_univ('equity', 'us_all')
    # univ.add_univ('equity', 'kr_sample')
    # univ.add_univ('equity', 'us_sample')
    # univ.add_univ('equity', 'us_all')

    pipe = Pipeline()
    pipe.add_pipeline('pipe_equity', universe=univ['equity'], item={'bm':bm, 'be':be})
    pipe.add_pipeline('pipe_equity2', universe=univ['equity'], item={'mktcap': mktcap})

    # sch_obj = Schedule('2017-01-01', '2017-03-01', type_='spec', days=[2, 6, 7])
    sch_obj = Schedule('2015-01-01', '2018-01-01', type_='end', freq_='m')

    begin_t = time.time()
    pipe.run_pipeline('pipe_equity2', schedule=sch_obj)
    end_t = time.time()

    data_be = pipe.get_item('pipe_equity', 'be')
    data_bm = pipe.get_item('pipe_equity', 'bm')
    data_mktcap = pipe.get_item('pipe_equity2', 'mktcap')
    return


