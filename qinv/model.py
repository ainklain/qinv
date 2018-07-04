import pandas as pd
import numpy as np
import time
from functools import partial
from qdata.datasource import TestingIO
from qinv.schedule import Schedule
from qinv.universe import Universe
from qinv.asset import Equity
from qdata.pipeline import Pipeline
__version__ = '1.0.0'


class Strategy:
    """
    data_be = fundamentals['be'].copy()
    x = Strategy(data_be)
    x.normalize()
    x.winsorize(0.01, True)
    make_order_table = x.make_order_table()
    bt = Testing(make_order_table)
    """
    def __init__(self, data):
        """
        :param data: index: (eval_d & infocode) / data: value_ 로 구성된 DataFrame
        """
        data.columns = pd.Index(['value_'])
        self.sch = sorted(list(set(data.index.get_level_values('eval_d'))))
        # self.sch = sorted(list(set(data['eval_d'])))
        self.data = data[~data['value_'].isna()]

    @staticmethod
    def _winsor_func(x, p, pct_winsor=False):
        # p: percent value if pct_winsor is True,
        #    absolute value if pct_winsor is False

        if pct_winsor is True:
            if p < 0.5:
                x[x > np.percentile(x, 100. * (1-p))] = np.percentile(x, 100. * (1-p))
                x[x < np.percentile(x, 100. * p)] = np.percentile(x, 100. * p)
            else:
                print('p should be lower than 0.5 if use pct_winsor.')
        else:
            x[x > p] = p
            x[x < -p] = -p

        return x

    def normalize(self):
        self.data['value_'] = self.data.groupby('eval_d')['value_'].transform(lambda x: (x - np.mean(x)) / np.std(x))

    def winsorize(self, p, pct_winsor=False):
        if p > 0:
            winsor_ftn = partial(Strategy._winsor_func, p=p, pct_winsor=pct_winsor)
            self.data['value_'] = self.data.groupby('eval_d')['value_'].transform(winsor_ftn)
        else:
            print('p(winsorize) should be positive value.')
            # if pct_winsor is False:
            #     self.data.loc[self.data['value_'] > winsorize, 'value_'] = winsorize
            #     self.data.loc[self.data['value_'] < -winsorize, 'value_'] = -winsorize
            # else:

    @staticmethod
    def _make_order_per_sch(x, q, wgt_method, long_short, reverse, q_type='rank', **kwargs):
        if q >= 0.5:
            print('q should be lower than 0.5. q will be set 0.499')
            q = 0.499
        elif q <= 0:
            print('q should be positive. q will be set 0.01')
            q = 0.01

        if q_type.lower() in ['rank']:
            idx_high = int(np.ceil(len(x) * (1.-q)))
            idx_low = int(len(x) * q)
            cond_high = (x >= sorted(x)[idx_high])
            cond_low = (x < sorted(x)[idx_low])
        else:
            q_high, q_low = (1. - q) * 100., q * 100.
            cond_high = (x >= np.percentile(x, q_high))
            cond_low = (x < np.percentile(x, q_low))

        if reverse is False:
            cond_long, cond_short = cond_high, cond_low
        else:
            cond_long, cond_short = cond_low, cond_high

        if wgt_method.lower() in ['eq', 'ew', 'equal']:
            wgt_long = 1. / sum(cond_long)
            wgt_short = 1. / sum(cond_short)

        wgt = np.zeros_like(x, dtype=float)
        if long_short.lower() in ['ls', 'l']:
            wgt[cond_high] = wgt_long
            pass
        if long_short.lower() in ['ls', 's']:
            wgt[cond_low] = -wgt_short
            pass

        return wgt

    def make_order_table(self, q=0.3, wgt_method='equal', long_short='LS', reverse=False, q_type='rank', **kwargs):
        print('q:{} / q_type:{} / reverse:{}'.format(q, q_type, reverse))
        make_order_ftn = partial(Strategy._make_order_per_sch,
                                 q=q,
                                 wgt_method=wgt_method,
                                 long_short=long_short,
                                 reverse=reverse,
                                 q_type=q_type, **kwargs)
        wgt_table = pd.DataFrame(columns=['wgt'])
        wgt_table['wgt'] = self.data.groupby('eval_d')['value_'].transform(make_order_ftn)
        wgt_table = wgt_table.reset_index()['eval_d', 'infocode', 'wgt']
        return wgt_table


class ModelDefault:
    def __init__(self, univ=None, sch=None):
        if univ is None:
            self.univ = Universe(**{'equity': ['kr_all']})
        else:
            self.univ = univ
        if sch is None:
            self.sch = Schedule('2000-01-01', '2016-01-01', type_='end', freq_='m')
        else:
            self.sch = sch

        self.equity_obj = Equity()
        self.equity_obj.initialize()
        self.pipe = Pipeline()

    def set_pipe(self, name, item, mode='load_or_run', chunksize=10000, table_owner=None):
        if mode == 'run':
            self.pipe.add(name, universe=self.univ, item=item)
            self.pipe.run(name, sch_obj=self.sch, mode=mode, chunksize=chunksize)
            return None
        is_loaded = self.pipe.load(name, table_owner=table_owner)
        if not is_loaded:
            self.pipe.add(name, universe=self.univ, item=item)
            self.pipe.run(name, sch_obj=self.sch, mode=mode, chunksize=chunksize)

    def set_pipe_by_info(self, **factor_info):
        pipe_nm = factor_info.pop('name')
        item_dict = factor_info.pop('item')
        self.set_pipe(pipe_nm, item_dict, **factor_info)

    def factor_write(self, winsorize=0.5, **factor_info):
        pipe_nm = factor_info['name']
        self.set_pipe(pipe_nm, factor_info['item'], **factor_info)
        factor_nm = list(factor_info['item'])[0]
        data = self.pipe.get_item(pipe_nm, item_id=factor_nm)

        my_model = Strategy(data)
        my_model.normalize(winsorize=winsorize)
        order_table = my_model.make_order_table(**factor_info)
        testing = Testing()
        result_bm_ls = testing.backtest(order_table)

        result_bm_ls.to_csv(r'txt//{}.csv'.format(factor_info['name']), header=True, index=None, sep=',', mode='w')
        return result_bm_ls


class Testing(TestingIO):
    def backtest(self, order_table, in_memory_test=False):
        if in_memory_test is False:
            self.store(order_table)
        else:
            print("In-memory calculation function will be added.")
            self.store(order_table)

        print("Successfully stored.")

        print("Calculating...")
        s = time.time()
        df = self.execute('backtest')
        e = time.time()
        print("Done! [{0:.2f}]".format(e-s))
        return df









"""
     select a.infocode, a.eval_d , e.eval_d, e.buy_d
		, exp(sum(log(1+isnull(y, 0.))) 
		over (	partition by a.infocode, a.eval_d 
				order by e.eval_d rows 
				between unbounded preceding and current row))-1 as a
	from (
		select eval_d, isnull(lead(eval_d, 1) over (order by eval_d), '9999-12-31') as fwd_d 
		from qinv..datetable 
		where eval_d between '2015-01-01' and '2018-01-01'  and  is_m_end=1
	) D
	join qpipe..test A
	on D.eval_d = A.eval_d
	join qinv..EquityTradeDate E
	on E.Eval_d > d.eval_d and E.Eval_d <= d.fwd_d
	and a.infocode = E.Infocode
	left join qinv..EquityReturnDaily R
	on e.Eval_d = r.marketdate and a.infocode = r.infocode
	order by a.infocode, a.eval_d, e.eval_d
        """


"""
SELECT  R.Infocode, R.MarketDate, count(y) over (
				PARTITION BY Infocode
              ORDER BY MArketDate 
              ROWS BETWEEN 20 PRECEDING AND CURRENT ROW ) as c
,
        STDEV(y)
           OVER ( 
				PARTITION BY Infocode
              ORDER BY MArketDate 
              ROWS BETWEEN 20 PRECEDING AND CURRENT ROW 
           ) * sqrt(250) AS s
FROM    qinv..EquityReturnDaily R
where infocode <= 100000
"""
