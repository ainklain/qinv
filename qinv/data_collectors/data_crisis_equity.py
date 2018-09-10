
from qinv.schedule import Schedule
from qinv.universe import Universe
from qinv.model import ModelDefault
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
from tqdm import tqdm
import pickle
from functools import partial


class TFModel(ModelDefault):
    def __init__(self, univ=None, sch=None, **kwargs):
        super().__init__(univ, sch, **kwargs)

    def set_data(self, name='cnn_test', mode='load_or_run'):
        item_dict = dict()
        if name == 'cnn_test':
            item_list = ['close_', 'close_adj', 'mktcap', 'be', 'gpa']

            for item in item_list:
                item_dict[item] = self.equity_obj[item]
            item_dict['bm'] = item_dict['be'] / item_dict['mktcap']

        elif name == 'crisis_data':
            item_list = ['close_', 'close_adj', 'mktcap',
                         'be', 'netsales', 'cogs', 'grossincome', 'netincome',
                         'depreciation', 'capitalexp', 'workingcap', 'totasset',
                         'totdebt', 'commsharesout', 'gpa', 'gmar', 'roa', 'roe']

            for item in item_list:
                item_dict[item] = self.equity_obj[item]

        elif name == 'fundamental_data':

            item_list = ['close_', 'close_adj', 'mktcap',
                         'be', 'netincome', 'netsales', 'cogs', 'sellinggeneralexp', 'opincome',
                         'stinvest', 'receivables', 'inventories', 'othercurasset', 'properties',
                         'otherasset', 'curdebt', 'accpayable', 'incometaxespayable', 'othercurdebt',
                         'totdebt']

            for item in item_list:
                item_dict[item] = self.equity_obj[item]

        factor_info = {'name': name,
                       'item': item_dict,
                       'mode': mode,
                       'chunksize': 15000, }

        self.set_pipe_by_info(**factor_info)


# (NetIncome + Depreciation - CapitalExp - (WorkingCap - WorkingCap_1)) / totAsset
# (NetSales - COGS) / NetSales
# (NetSales - COGS) / TotAsset
# (Depreciation - (WorkingCap - WorkingCap_1)) / totAsset
# NetIncome / TotAsset
# NetIncome / BE
#
# -log(TOTDebt / TOTDebt_1)
# -log(CommSharesOut / CommSharesOut_1)
#
# (GrossIncome - GrossIncome_5) / totAsset_5
# GPA - GPA_5
# GMAR - GMAR_5
# ROA - ROA_5
# ROE - ROE_5
#
# beta, iVol
# totDebt / totAsset


m = TFModel()
m = TFModel(univ=Universe(**{'equity': ['us_all']}),
            sch=Schedule('1985-01-01', '2016-12-31', type_='end', freq_='m'))

data_name = 'crisis_data'
# data_name = 'fundamental_data'
# m.set_data(data_name)
m.pipe.load(data_name)
# m.pipe.add_item(data_name, {
#     'gmar': m.equity_obj['gmar'],}, sudo=True)
# m.pipe.run(data_name, mode='update')

# m.pipe.add_item('fundamental_data', {
# #     'sellinggeneralexp': sellinggeneralexp,
#     'receivables': receivables,
#     'othercurasset': othercurasset,
#     'otherasset': otherasset,
#     'accpayable': accpayable,
#     'othercurdebt': othercurdebt,
# }, sudo=True)
# m.pipe.run('fundamental_data', mode='update')


index_columns = ['infocode', 'eval_d']
# fundamental data (dictionary format)
base_fundamental = 'netsales'
fundamental_list = ['be', 'netsales', 'cogs', 'grossincome', 'netincome',
                    'depreciation', 'capitalexp', 'workingcap', 'totasset',
                    'totdebt', 'commsharesout', 'gpa', 'gmar', 'accrual',
                    'roa', 'roe']


fundamentals = dict()
for name in fundamental_list:
    fundamentals[name] = m.pipe.get_item(data_name, item_id=name)
    fundamentals[name].drop(labels=['item_nm'], axis=1, inplace=True)
    fundamentals[name].columns.values[-1] = name
    fundamentals[name].set_index(index_columns, inplace=True)


# monthly prices and returns
close_ = m.pipe.get_item(data_name, item_id='close_')

close_adj = m.pipe.get_item(data_name, item_id='close_adj')
close_adj.drop(labels='item_nm', axis=1, inplace=True)
close_adj.set_index(index_columns, inplace=True)

mktcap = m.pipe.get_item(data_name, item_id='mktcap')
mktcap.drop(labels='item_nm', axis=1, inplace=True)
mktcap.set_index(index_columns, inplace=True)

returns = pd.DataFrame(index=close_adj.index)
return_months = [1, 3, 6, 9, 12]

for i_mon, mon in enumerate(sorted(return_months)):
    if i_mon == 0:
        # 가장 짧은 주기 기준으로 null 값 제거
        base_item = mon
    return_ftn = partial(pd.Series.pct_change, periods=mon)
    returns['y_{}m'.format(mon)] = close_adj.groupby('infocode')['value_'].transform(return_ftn)

# returns = returns[~returns['y_{}m'.format(base_item)].isnull()]   # base_item에 해당하는 month  기준으로 null제거
returns = returns[np.sum(returns.isnull(), axis=1) == 0] # null 하나라도 있으면 전부 제거

sch = sorted(list(set(returns.index.get_level_values('eval_d'))))

from qdata.dbmanager import SqlManager
sqlm = SqlManager()
sqlm.set_db_name('qinv')


fundamental_data = None
fundamental_label = None
for i in range(48, len(sch)-12):
    if i % 20 == 0:
        print(i)

    t_next = sch[i+12]
    t = sch[i]

    # universe
    cond_sch = (close_['eval_d'] == t)
    univ_code = close_[close_['eval_d'] == t][['infocode']]
    for j in range(1, 5):
        cond_sch = cond_sch | (close_['eval_d'] == sch[i - 12 * j])

    cond_ex_coin = np.array(close_['value_'] >= 5)
    univ_t = pd.merge(univ_code, close_[cond_sch & cond_ex_coin][index_columns]).set_index(index_columns)
    # for t_next
    cond_sch_next = (close_['eval_d'] == t_next)
    univ_t_next = pd.merge(univ_code, close_[cond_sch_next & cond_ex_coin][index_columns]).set_index(index_columns)

    # fundamental data for t

    fundamentals_t = pd.DataFrame(index=univ_t.index)
    fundamentals_t_next = pd.DataFrame(index=univ_t_next.index)
    # mktcap_t = pd.merge(univ_t, mktcap.loc[mktcap.index.isin([t], level='eval_d')], how='left', on=index_columns)
    # for name in fundamental_list:
    #     fundamentals_t[name] = pd.merge(univ_t, fundamentals[name], how='left', on=index_columns).values \
    #                            / mktcap_t.fillna(method='bfill').values
    #     fundamentals_t_next[name] = pd.merge(univ_t_next, fundamentals[name], how='left', on=index_columns) \
    #                                 / pd.merge(univ_t_next, mktcap_t.dropna(), how='left', on='infocode').values

    mktcap_t = pd.merge(univ_t, mktcap.loc[mktcap.index.isin([t], level='eval_d')]
                        , how='left', on=index_columns).fillna(method='bfill')
    for name in fundamental_list:
        fundamentals_t[name] = pd.merge(univ_t, fundamentals[name], how='left', on=index_columns).values / mktcap_t.values
        fundamentals_t_next[name] = pd.merge(univ_t_next, fundamentals[name], how='left', on=index_columns)

    # return data for t
    mom_t = pd.merge(univ_t, returns, on=index_columns)
    mom_t = mom_t.groupby('eval_d').rank(ascending=False, pct=True)

    # all data for t
    all_t = pd.merge(mom_t, fundamentals_t, how='left', on=index_columns)
    all_t = all_t[~all_t[base_fundamental].isnull()]
    all_t = all_t.groupby(['infocode']).fillna(method='ffill')
    all_t_std = (all_t - all_t.mean(axis=0)) / all_t.std(axis=0)

    x = all_t_std.reset_index().pivot(index='infocode', columns='eval_d').fillna(value=0)
    x.columns = x.columns.droplevel('eval_d')
    len(x.columns)

    y = fundamentals_t_next.reset_index().pivot(index='infocode', columns='eval_d')
    y = y[np.sum(~y.isnull(), axis=1) > 0].fillna(value=0)
    y.columns = y.columns.droplevel('eval_d')

    data_t = pd.merge(x, y, on='infocode').iloc[:, :len(x.columns)]
    label_t = pd.merge(x, y, on='infocode').iloc[:, len(x.columns):]


    if len(data_t) == 0:
        # 데이터 없음
        continue

    if len(data_t.columns) != (len(label_t.columns) + 5) * 5:
        print("fail: {}".format(t))
        continue

    if fundamental_data is None:
        fundamental_data = data_t.copy()
    else:
        fundamental_data = np.concatenate([fundamental_data, data_t])

    if fundamental_label is None:
        fundamental_label = label_t.copy()
    else:
        fundamental_label = np.concatenate([fundamental_label, label_t])


f = open('data/crisis.pkl', 'wb')
pickle.dump({'data': fundamental_data, 'label': fundamental_label}, f)

