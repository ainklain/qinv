import pandas as pd
import numpy as np
import time

from datasource import Equity, Schedule, Universe, Item
from pipeline import Pipeline
from model import Model, BackTesting

equity_obj = Equity()
equity_obj.initialize()
mktcap = equity_obj['mktcap']
be = equity_obj['be']
bm = be / mktcap

univ = Universe()
univ.add_univ('equity', 'us_all')


# new pipe
pipe = Pipeline()
pipe.add_pipeline('pipe_equity', universe=univ, item={'bm': bm, 'be': be})
pipe.add_pipeline('pipe_equity2', universe=univ, item={'mktcap': mktcap})
pipe.load_pipeline(name='pipe_equity')

sch_obj = Schedule('2015-01-01', '2018-01-01', type_='end', freq_='m')
pipe.run_pipeline('pipe_equity2', schedule=sch_obj)

# pipe load
pipe = Pipeline('pipe_equity')
pipe = Pipeline('pipe_equity2')
pipe.run_pipeline('pipe_equity')


data_be = pipe.get_item('pipe_equity', 'be')
data_bm = pipe.get_item('pipe_equity', 'bm')
data_mktcap = pipe.get_item('pipe_equity2', 'mktcap')

x = Model(data_mktcap)
x.normalize(winsorize=0.5)
order_signal = x.make_order_signal(min_n_of_stocks=2)
bt = BackTesting(sch_obj)

