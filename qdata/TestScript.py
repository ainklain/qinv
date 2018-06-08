from qinv.schedule import Schedule
from qinv.universe import Universe
from qinv.asset import Equity
from qdata.pipeline import Pipeline
from qinv.model import Model, Testing

equity_obj = Equity()
equity_obj.initialize()
mktcap = equity_obj['mktcap']
be = equity_obj['be']
bm = be / mktcap
gpa = equity_obj['gpa']
# Universe 생성
# 방법1
univ = Universe()
univ.add_univ('equity', 'us_all')
# 방법2
univ_test = {'equity': ['us_all', 'kr_all']}
univ = Universe(**univ_test)

# new pipe
pipe = Pipeline()
pipe.add('pipe_equity', universe=univ, item={'bm': bm, 'be': be})
pipe.add('pipe_equity2', universe=univ, item={'mktcap': mktcap})
pipe.add('pipe_gpa', universe=univ, item={'gpa': gpa})
pipe.add('pipe_be', universe=univ, item={'be': be})
pipe.add('pipe_bm', universe=univ, item={'bm': bm})
pipe.load(name='pipe_equity')
pipe.add_item('pipe_equity', {'gpa': gpa})

sch_obj = Schedule('2000-01-01', '2018-01-01', type_='end', freq_='m')
# DB에 같은 테이블 있어도 강제로 저장옵션
# pipe.run_pipeline('pipe_equity', schedule=sch_obj, store_db=True)
# pipe.run_pipeline('pipe_gpa', schedule=sch_obj, store_db=True)
# pipe.run_pipeline('pipe_be', schedule=sch_obj, store_db=True, chunksize=1000)
# pipe.run_pipeline('pipe_bm', schedule=sch_obj, store_db=True, chunksize=1000)
# pipe.get_code('pipe_equity', schedule=sch_obj)
# pipe load
pipe = Pipeline('pipe_equity')
pipe = Pipeline('pipe_equity2')
pipe.run('pipe_equity')


# data_be = pipe.get_item('pipe_equity', 'be')
data_bm = pipe.get_item('pipe_equity', item_id='bm')
# data_mktcap = pipe.get_item('pipe_equity2', 'mktcap')

x = Model(data_bm)
x.normalize(winsorize=0.5)
order_table = x.make_order_table(min_n_of_stocks=2)
testing = Testing()
result_bm_ls = testing.backtest(order_table)

result_bm_ls.to_csv(r'txt//final.csv', header=True, index=None, sep=',', mode='w')
