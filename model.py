import pandas as pd
import numpy as np
import time
import socket
import settings
from dbmanager import SqlManager

from datasource import Equity, Schedule, Universe
from pipeline import Pipeline
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

        if winsorize > 0:
            self.data.loc[self.data['value_'] > winsorize, 'value_'] = winsorize 
            self.data.loc[self.data['value_'] < -winsorize, 'value_'] = -winsorize

    def make_order_table(self, q=0.3, weight='equal', long_short='LS', reverse=False, min_n_of_stocks=10, **kw):
        if q >= 1:
            q = 1.
        elif q <= 0:
            q = 0.

        if reverse is False:
            q_long = (1. - q) * 100.
            q_short = q * 100.
        else:
            q_long = q * 100.
            q_short = (1. - q) * 100.

        sch_skipped = list()
        wgt_table = pd.DataFrame(columns=['eval_d', 'infocode', 'wgt'])
        for sch_ in self.sch:
            cond_sch = np.array(self.data['eval_d'] == sch_)
            cond_long = np.array(self.data[cond_sch]['value_']
                                 >= np.percentile(self.data[cond_sch]['value_'], q_long))
            cond_short = np.array(self.data[cond_sch]['value_']
                                  <= np.percentile(self.data[cond_sch]['value_'], q_short))

            # 포트폴리오 구성 최소 종목수
            if sum(cond_long) < min_n_of_stocks or sum(cond_short) < min_n_of_stocks:
                sch_skipped.append(sch_)
                continue

            # weighting scheme 변경
            if weight.lower() in ['equal', 'eq', 'ew']:
                wgt_long = 1. / sum(cond_long)
                wgt_short = 1. / sum(cond_short)
            elif weight.lower() in ['rank']:
                pass
            elif weight.lower() in ['value', 'vw']:
                pass
            elif weight.lower() in ['volatility', 'vol']:
                pass
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

    def set_pipe(self, pipe_nm, item_dict):
        is_loaded = self.pipe.load_pipeline(name=pipe_nm)
        if not is_loaded:
            self.pipe.add_pipeline(pipe_nm, universe=self.univ, item=item_dict)
            self.pipe.run_pipeline(pipe_nm, schedule=self.sch, store_db=False, chunksize=1000)

    def factor_write(self, winsorize=0.5, **factor_info):
        pipe_nm = 'pipe_' + factor_info['name']
        self.set_pipe(pipe_nm, factor_info['item'])
        factor_nm = list(factor_info['item'])[0]
        data = self.pipe.get_item(pipe_nm, factor_nm)

        my_model = Model(data)
        my_model.normalize(winsorize=winsorize)
        order_table = my_model.make_order_table(**factor_info)
        testing = Testing()
        result_bm_ls = testing.backtest(order_table)

        result_bm_ls.to_csv(r'txt//{}.csv'.format(factor_info['name']), header=True, index=None, sep=',', mode='w')



class MyModel(Model):
    def __init__(self):
        pass


class TestingIO:
    def __init__(self):
        self.sqlm = SqlManager()
        self.sqlm.set_db_name('qpipe')

    @staticmethod
    def _get_table_id(process_nm):
        my_ip = socket.gethostbyname(socket.gethostname())
        if my_ip in settings.ip_class.keys():
            table_id = process_nm + '_' + settings.ip_class[my_ip]
        else:
            table_id = process_nm + '_' + 'unknown'
        return table_id

    def store_order(self, order_table, process_nm='backtest'):
        import time
        sqlm = self.sqlm

        table_id = self._get_table_id(process_nm)

        sqlm.db_execute("""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME=N'{table_id}')
        BEGIN
        create table {table_id} (eval_d date, infocode int, wgt float
        primary key(eval_d, infocode)
        )
        END
        ELSE BEGIN
        truncate table {table_id}
        END
        """.format(table_id=table_id))
        print('Table {} created.'.format(table_id))

        s = time.time()
        sqlm.db_insert(order_table, table_id, fast_executemany=True)
        print('Inserted.')
        e = time.time()
        print("calculation time: {0:.2f} sec".format(e - s))

    def execute(self, process_nm='backtest'):
        if process_nm in ['backtest']:
            table_id = self._get_table_id(process_nm)
            sql_ = """            
select a.eval_d, 100 + sum(cum_w) as w, 100 + sum(cum_y) as y,  (100 + sum(cum_y)) / (100 + sum(cum_w)) - 1 as y
from (
select  a.infocode, a.eval_d as calc_d, e.eval_d, y
    , 100 * a.wgt * exp(sum(log(1+isnull(y, 0.))) 
        over (	partition by a.infocode, a.eval_d 
                order by e.eval_d rows 
                between unbounded preceding and current row)) as cum_y
    , 100 * a.wgt * isnull(exp(sum(log(1+isnull(y, 0.))) 
        over (	partition by a.infocode, a.eval_d 
                order by e.eval_d rows 
                between unbounded preceding and 1 preceding)), 1) as cum_w
    from (
        select eval_d, isnull(lead(eval_d, 1) over (order by eval_d), dateadd(day, 31, eval_d)) as fwd_d
            from qpipe..{table_id}		
            group by eval_d
    ) D
    join (select * from qpipe..{table_id} A where wgt >= 0) a
    on D.eval_d = A.eval_d	
    join qinv..EquityTradeDate E
    on a.infocode = E.Infocode and E.Eval_d > d.eval_d and E.Eval_d <= d.fwd_d
    left join qinv..EquityReturnDaily R
    on a.infocode = r.infocode and e.Eval_d = r.marketdate 
) A
group by a.eval_d
order by a.eval_d
            """.format(table_id=table_id)

            df = self.sqlm.db_read(sql_)
            return df




class Testing(TestingIO):
    def backtest(self, order_table, in_memory_test=False):
        if in_memory_test is False:
            self.store_order(order_table)
        else:
            print("In-memory calculation function will be added.")
            self.store_order(order_table)

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
