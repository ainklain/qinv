import pandas as pd
import numpy as np
import time
from qdata.datasource import TestingIO
from qinv.schedule import Schedule
from qinv.universe import Universe
from qinv.asset import Equity
from qdata.pipeline import Pipeline
__version__ = '1.0.0'


class Model:
    """
    x = Model(data_mktcap)
    x.normalize(winsorize=0.5)
    order_signal = x.make_order_signal(min_n_of_stocks=2)
    bt = BackTesting(sch_obj)
    """
    def __init__(self, data):
        """
        :param data: eval_d / infocode / value_ 로 구성된 DataFrame 
        """
        self.sch = sorted(list(set(data['eval_d'])))
        self.data = data[~data['value_'].isna()]

    def normalize(self, winsorize=0.0):
        for sch_ in self.sch:
            cond_sch = np.array(self.data['eval_d'] == sch_)
            arr = np.array(self.data.loc[cond_sch, 'value_'])
            self.data.loc[cond_sch, 'value_'] = (arr - np.mean(arr)) / np.std(arr)

        if winsorize > 0 or winsorize is None:
            self.data.loc[self.data['value_'] > winsorize, 'value_'] = winsorize
            self.data.loc[self.data['value_'] < -winsorize, 'value_'] = -winsorize

    def make_order_table(self, q=0.3, weight='equal', long_short='LS', reverse=False, min_n_of_stocks=10, **kw):
        print('q:{} / reverse:{}'.format(q, reverse))

        if q >= 0.5:
            q = 0.499
        elif q <= 0:
            q = 0.

        q_high = int((1. - q) * 100.)
        q_low = int(q * 100.)

        sch_skipped = list()
        wgt_table = pd.DataFrame(columns=['eval_d', 'infocode', 'wgt'])
        for sch_ in self.sch:
            cond_sch = np.array(self.data['eval_d'] == sch_)
            cond_high = np.array(self.data[cond_sch]['value_']
                                 >= np.percentile(self.data[cond_sch]['value_'], q_high)+0.0001)
            cond_low = np.array(self.data[cond_sch]['value_']
                                <= np.percentile(self.data[cond_sch]['value_'], q_low)-0.0001)
            # cond_high = cond_low 인 종목의 경우 primary key 에러 발생(L에도 포함, S에도 포함. 0.0001로 방지)

            if reverse is False:
                cond_long = cond_high
                cond_short = cond_low
            else:
                cond_long = cond_low
                cond_short = cond_high

            # 포트폴리오 구성 최소 종목수
            if sum(cond_long) < min_n_of_stocks or sum(cond_short) < min_n_of_stocks:
                sch_skipped.append(sch_)
                continue

            # weighting scheme 변경
            if weight.lower() in ['equal', 'eq', 'ew']:
                wgt_long = 1. / sum(cond_long)
                wgt_short = 1. / sum(cond_short)
            # elif weight.lower() in ['rank']:
            #     pass
            # elif weight.lower() in ['value', 'vw']:
            #     pass
            # elif weight.lower() in ['volatility', 'vol']:
            #     pass
            else:
                wgt_long = 1. / sum(cond_long)
                wgt_short = 1. / sum(cond_short)

            # Long portfolio
            if long_short.lower() in ['ls', 'l']:
                wgt_table_long = pd.DataFrame(data={
                    'eval_d': self.data[cond_sch][cond_long]['eval_d'],
                    'infocode': self.data[cond_sch][cond_long]['infocode'],
                    'wgt': wgt_long})
                wgt_table = pd.concat([wgt_table, wgt_table_long], ignore_index=True)

            # Short portfolio
            if long_short.lower() in ['ls', 's']:
                wgt_table_short = pd.DataFrame(data={
                    'eval_d': self.data[cond_sch][cond_short]['eval_d'],
                    'infocode': self.data[cond_sch][cond_short]['infocode'],
                    'wgt': -wgt_short})
                wgt_table = pd.concat([wgt_table, wgt_table_short], ignore_index=True)

        return wgt_table


class ModelDefault:
    def __init__(self, univ=None, sch=None, **kwargs):
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

    def set_pipe(self, pipe_nm, item_dict, store_result=False, overwrite_pipe=False, chunksize=1000, **kwargs):
        is_loaded = self.pipe.load(name=pipe_nm, overwrite_pipe=overwrite_pipe)
        if not is_loaded:
            self.pipe.add(pipe_nm, universe=self.univ, item=item_dict)
            self.pipe.run(pipe_nm, schedule=self.sch, store_result=store_result, chunksize=chunksize)

    def set_pipe_by_info(self, **factor_info):
        pipe_nm = factor_info['name']
        item_dict = factor_info['item']
        self.set_pipe(pipe_nm, item_dict, **factor_info)

    def factor_write(self, winsorize=0.5, **factor_info):
        pipe_nm = factor_info['name']
        self.set_pipe(pipe_nm, factor_info['item'], **factor_info)
        factor_nm = list(factor_info['item'])[0]
        data = self.pipe.get_item(pipe_nm, item_id=factor_nm)

        my_model = Model(data)
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
