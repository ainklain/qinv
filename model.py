import pandas as pd
import numpy as np

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

    def make_order_signal(self, q=0.3, weight='equal', long_short='LS', reverse=False, min_n_of_stocks=10):
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


class MyModel(Model):
    def __init__(self):
        pass


class BackTesting:
    def __init__(self, schedule):
        self.schedule = schedule.code.lower().replace(
            "eval_d as eval_d", "eval_d, isnull(lead(eval_d, 1) over (order by eval_d), '9999-12-31') as fwd_d")

    def backtest(self, order_signal):
        from dbmanager import SqlManager
        import socket
        import settings
        import time
        sqlm = SqlManager()
        sqlm.set_db_name('qpipe')

        my_ip = socket.gethostbyname(socket.gethostname())
        if my_ip in settings.ip_class.keys():
            table_id = 'backtest_' + settings.ip_class[my_ip]
        else:
            table_id = 'backtest_' + 'unknown'

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

        sqlm.db_insert(order_signal, table_id, True)
        print('Inserted.')

        s = time.time()
        """
        select a.eval_d, sum(a)
from (
        
             select a.infocode, a.eval_d as calc_d, e.eval_d
        		,  a.wgt * exp(sum(log(1+isnull(y, 0.))) 
        		over (	partition by a.infocode, a.eval_d 
        				order by e.eval_d rows 
        				between unbounded preceding and current row))-1 as a
        	from ({}) D
        	join qpipe..{} A
        	on D.eval_d = A.eval_d
        	join qinv..EquityTradeDate E
        	on E.Eval_d > d.eval_d and E.Eval_d <= d.fwd_d
        	and a.infocode = E.Infocode
        	left join qinv..EquityReturnDaily R
        	on e.Eval_d = r.marketdate and a.infocode = r.infocode
        	
        	
) a
group by eval_d
        """.format(self.schedule, table_id)

        e = time.time()
        print("calculation time: {} sec".format(e - s))








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
