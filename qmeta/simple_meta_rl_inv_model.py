# issue :   meta_batch_size 로 accuracies를 계산하는데 data_generator에서 산출되는 인풋개수가 support/query로 나뉘다보니 개수 안맞음.
#           train / valid set 간 개수 잘 맞춰야할것

from qdata.dbmanager import SqlManager

import argparse
import csv
import datetime
import numpy as np
import pandas as pd
import pickle
import random
import tensorflow as tf
import matplotlib.pyplot as plt

import gym

from tensorflow.contrib.layers.python import layers as tf_layers

parser = argparse.ArgumentParser()
parser.add_argument('--datasource', default='momentum')


class Argument:
    def __init__(self):
        self.datasource = 'momentum'
        self.idx_list = ['MSCIWORLD', 'MSCIEM', 'KOSPI200', 'KOSDAQ150', 'SPX500', 'NASDAQ100', 'RUSSELL2000',
                    'CSI300', 'HSCE', 'TOPIX100', 'MSCIUSREITTR', 'USDKRW', 'GSCIGOLD', 'KISCOMPBONDCALL']
        # self.idx_list = ['MSCIWORLD', 'KISCOMPBONDCALL']

        self.num_timesteps = 12
        self.stop_grad = True
        self.meta_batch_size = 24
        self.update_batch_size = 1      # K for K-shot learning
        self.train = True
        # self.test_set = False
        self.train_update_batch_size = -1
        self.train_update_lr = -1
        self.update_lr = 1e-3
        self.num_classes = 2
        self.num_updates = 1
        self.num_filters = 64
        self.max_pool = False
        self.baseline = None
        self.norm = 'None'
        self.resume = True
        self.logdir = '/tmp/data'
        self.test_iter = -1
        self.pretrain_iterations = 0
        self.metatrain_iterations = 20000
        self.log = True

args = Argument()


def factor_ls_txt():
    file_nm = 'factor_data.txt'
    df = pd.read_csv(file_nm, index_col=0, sep='\t')

    df.columns = [i.lower() for i in df.columns]
    df.set_index('eval_d', inplace=True)
    df = df[df.isna().sum(axis=1) == 0]
    etf_id = list(df.columns)
    marketdate = list(df.index.unique())
    history = df.values


def factor_history():
    from qdata.dbmanager import SqlManager
    sql_ = """    	
		select eval_d, mkt_rf, smb, hml, rmw, wml, call_rate, kospi, mom, beme, gpa, usdkrw
			from (
				select return_date, mkt_rf, smb, hml, rmw, wml, call_rate, mkt
					from passive..factor_timeseries_wise
					where return_date >= '2005-02-02'
			) a
			join (
				select * 
					from  (
					select idxcd, d.eval_d, round(clsprc / lag(clsprc, 1) over (partition by idxcd order by base_d) - 1, 8) as value_
						from (
							select eval_d
							from qinv..DateTableGlobal
							where region = 'KR'
							and eval_d = work_d
							and eval_d >= '2005-02-01'
						) D
						join wms..indexdata a
						on d.eval_d = convert(date, a.base_d)
						where idxcd in ('KOSPI','MOM','BEME','GPA','USDKRW','KISCOMPBONDCALL')
						and base_d >= '20050201'
					) A
					pivot (
						min(value_)
						for idxcd in ([KOSPI],[MOM],[BEME],[GPA],[USDKRW],[KISCOMPBONDCALL])
					) B
			) b
			on a.return_date = b.eval_d
			order by a.return_date, b.eval_d
    """
    sqlm = SqlManager()
    df = sqlm.db_read(sql_)
    df.columns = [i.lower() for i in df.columns]
    df.set_index('eval_d', inplace=True)
    df = df[df.isna().sum(axis=1) == 0]
    factor_id = list(df.columns)
    marketdate = list(df.index.unique())

    history = df.values

    # history = df_to_arr[~np.isnan(np.sum(df_to_arr, axis=1)), :]

    return df, history, factor_id, marketdate


def factor_etf_csv():
    file_nm = 'factor_etf.csv'
    df = pd.read_csv(file_nm, index_col=0)

    df.columns = [i.lower() for i in df.columns]
    df.set_index('eval_d', inplace=True)
    df = df[df.isna().sum(axis=1) == 0]
    etf_id = list(df.columns)
    marketdate = list(df.index.unique())
    history = df.values

    return history, etf_id, marketdate


def factor_history_csv():
    file_nm = 'data_for_metarl.csv'
    df = pd.read_csv(file_nm, index_col=0)

    df.columns = [i.lower() for i in df.columns]
    df = df[df.isna().sum(axis=1) == 0]
    # columns = list(df.columns)
    # marketdate = list(df.index.unique())
    # history = df.values

    return df # , history, columns, marketdate



class PortfolioSim(object):
    def __init__(self, asset_list, macro_list=None, steps=250, trading_cost=1e-3):
        self.trading_cost = trading_cost
        self.steps = steps
        self.step = 0

        self.assets_return_df = pd.DataFrame(columns=asset_list)
        self.macros_return_df = None
        if macro_list is not None:
            self.macros_return_df = pd.DataFrame(columns=macro_list)

        self.actions = np.zeros([self.steps, len(asset_list)])
        self.navs = np.ones(self.steps)
        self.assets_nav = np.ones([self.steps, len(asset_list)])
        self.positions = np.zeros([self.steps, len(asset_list)])
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros([self.steps, len(asset_list)])
        self.rewards_history = np.ones(self.steps)

    def _reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.assets_nav.fill(1)
        self.rewards_history.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)

        self.assets_return_df = self.assets_return_df.iloc[0:0]
        if self.macros_return_df is not None:
            self.macros_return_df = self.macros_return_df.iloc[0:0]

    def _step(self, actions, assets_return, macros_return=None):
        eps = 1e-8

        if self.step == 0:
            last_pos = np.zeros(len(actions))
            last_nav = 1.
            last_asset_nav = np.ones(len(actions))
        else:
            last_pos = self.positions[self.step - 1, :]
            last_nav = self.navs[self.step - 1]
            last_asset_nav = self.assets_nav[self.step - 1, :]

        self.assets_return_df.loc[self.step] = assets_return
        if macros_return is not None:
            self.macros_return_df.loc[self.step] = macros_return

        self.actions[self.step, :] = actions

        self.positions[self.step, :] = ((assets_return + 1.) * actions) / (np.dot((assets_return + 1.), actions) + eps)
        self.trades[self.step, :] = actions - last_pos

        trade_costs_pct = np.sum(abs(self.trades[self.step, :])) * self.trading_cost
        self.costs[self.step] = trade_costs_pct
        instant_reward = (np.dot((assets_return + 1.), actions) - 1.) - self.costs[self.step]

        if self.step != 0:
            self.navs[self.step] = last_nav * (1. + instant_reward)
            self.stk_nav[self.step, :] = last_asset_nav * (1. + assets_return)

        if (self.navs[self.step] == 0) | (self.navs[self.step] < np.max(self.navs) * 0.9):
            done = True
            winning_reward = -1
        elif self.step == self.steps:
            done = True
            if self.navs[self.step - 1] >= (1 + 0.05 * (self.step / 250)):      # 1년 5 % 이상 (목표)
                winning_reward = 1
            else:
                winning_reward = -1
        else:
            winning_reward = 0

        total_reward = 0.1 * instant_reward + 0.9 * winning_reward
        self.rewards_history[self.step] = total_reward

        info = {'instant_reward': instant_reward,
                'winning_reward': winning_reward,
                'nav': self.navs[self.step],
                'costs': self.costs[self.step]}

        self.step += 1
        return total_reward, info, done

    def export(self):
        exported_data = dict()
        exported_data['last_step'] = self.step
        exported_data['asset_returns_df'] = self.assets_return_df
        exported_data['macro_returns_df'] = self.macros_return_df
        exported_data['navs'] = self.navs
        exported_data['positions'] = self.positions
        exported_data['costs'] = self.costs
        exported_data['actions'] = self.actions

        return exported_data


class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, asset_df, macro_df=None, trading_cost=0.0020, window_length=250, is_training=True):
        super().__init__()
        self.window_length = window_length
        self.trading_cost = trading_cost

        self._setup(asset_df, macro_df)

    def _setup(self, asset_df, macro_df=None):
        self.asset_list = list(asset_df.columns)
        if macro_df is not None:
            self.macro_list = list(macro_df.columns)
            assert asset_df.shape[0] == macro_df.shape[0], 'length of asset_df should be same as that of macro_df.'
        else:
            self.macro_list = []

        self.sim = PortfolioSim(self.asset_list, self.macro_list, steps=250, trading_cost=self.trading_cost)

        self.action_space = gym.spaces.Box(0, 1, shape=(len(self.asset_list), ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(self.window_length, len(self.asset_list) + len(self.macro_list)),
                                                dtype=np.float32)

        self._data = pd.concat([asset_df, macro_df], axis=1)

    def step(self, actions):
        return self._step(actions)

    def _step(self, actions, eps=1e-8):
        # np.testing.assert_almost_equal(actions.shape, (self.action_space.shape,))


        action = np.clip(actions, 0.0, 1)
        weights = np.zeros_like(action)
        weights[action > 0] = action[action > 0] / (np.sum(action[action > 0]) + eps)
        weights[-1] = 1 - np.sum(weights[:-1])
        # weights[action < 0] = action[action < 0] / (np.sum(action[action < 0]) + eps)


        assert (weights >= (-1 - 1e-6) * (weights <= (1 + 1e-6))).all()
        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3,
                                       err_msg='\n[err msg] \nsum of weights: {}\naction:{}'.format(np.sum(weights), action))
        # np.testing.assert_almost_equal(np.sum(weights), 0.0, 3)

        obs, done1, ground_truth_obs = self.src._step()

        y1 = obs[-1, :]
        reward, info, done2 = self.sim._step(weights, y1)
        # reward = reward - np.sum(weights == 0) * 0.01
        self.infos.append(info)

        obs = np.expand_dims(obs, -1)
        # print('reward:{} , info:{}'.format(reward, info))
        return obs, reward, done1 or done2, info


    def reset(self):
        return self._reset()

    def _reset(self):
        pass

    def render(self, mode='human', close=False):
        return self._render(mode=mode,  close=close)

    def _render(self, mode='human', close=False):
        pass


def dateadd(base_d, freq_='D', added_nums=0):
    # date_dt= datetime.datetime.strptime(base_d, '%Y-%m-%d')
    from dateutil.relativedelta import relativedelta
    from dateutil.parser import parse
    date_dt = parse(base_d)

    if freq_.lower() in ['y', 'year']:
        # date_dt = date_dt.replace(year=date_dt.year + added_nums)
        date_dt = date_dt + relativedelta(years=added_nums)
    elif freq_.lower() in ['m', 'month']:
        date_dt = date_dt + relativedelta(months=added_nums)
    elif freq_.lower() in ['w', 'week']:
        date_dt = date_dt + relativedelta(weeks=added_nums)
    else:
        date_dt = date_dt + relativedelta(days=added_nums)

    date_ = date_dt.strftime('%Y-%m-%d')
    return date_


class Episodes(object):
    def __init__(self, gamma=0.95):
        self.gamma = gamma





class EnvSampler(object):
    def __init__(self, envs_list, batch_size):
        self.envs_list = envs_list
        self.batch_size = batch_size

    def sample(self, policy, params=None, gamma=0.95):
        pass


    def sample_envs(self, base_d, num_envs):
        date_ = dateadd(base_d, 'm', -1)
        idx_ = self.envs_list.index(date_)
        # envs = random.sample(self.envs_list[:idx_], num_envs)
        envs = np.random.choice(self.envs_list[:idx_], size=num_envs, replace=False)
        return envs



class MyPolicy:
    def __init__(self, dim_input, dim_output, dim_hidden=[]):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.num_layers = len(dim_hidden) + 1

    def construct_weights(self, scope_nm=None):
        if scope_nm is not None:
            with tf.variable_scope(scope_nm):
                weights = {}
                weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
                weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
                for i in range(1, len(self.dim_hidden)):
                    weights['w' + str(i + 1)] = tf.Variable(
                        tf.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
                    weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
                weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
                    tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
                weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward(self, input_data, weights, reuse=False):
        input_data = tf.reshape(input_data, [-1, self.dim_input])
        hidden = normalize(tf.matmul(input_data, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse,
                           scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i + 1))
        out = tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights[
            'b' + str(len(self.dim_hidden) + 1)]
        return out






class MetaLearner(object):
    def __init__(self, sampler, policy, baseline, gamma=0.95, fast_lr=0.5, tau=1.0):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau

    def inner_loss(self, episodes, params=None):

        pass


    def abc(self):
        self.policy.forward()

    def construct_model(self):
        self.support_x = tf.placeholder(tf.float32, shape=[None, 1, self.dim_input], name='support_x')
        self.support_y = tf.placeholder(tf.float32, shape=[None, 1, self.dim_output], name='support_y')
        self.query_x = tf.placeholder(tf.float32, shape=[None, 1, self.dim_input], name='query_x')
        self.query_y = tf.placeholder(tf.float32, shape=[None, 1, self.dim_output], name='query_y')

            num_updates = self.test_num_updates

            def task_metalearn(input_data, reuse=True):
                support_x, support_y, query_x, query_y = input_data
                task_query_preds, task_query_losses = [], []

                if self.classification:
                    task_query_accs = []

                task_support_pred = self.forward(support_x, weights, reuse=reuse)
                task_support_loss = self.loss_func(task_support_pred, support_y)

                grads = tf.gradients(task_support_loss, list(weights.values()))
                if args.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]

                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(),
                                        [weights[key] - self.train_lr * gradients[key] for key in weights.keys()]))

                query_pred = self.forward(query_x, fast_weights, reuse=True)
                task_query_preds.append(query_pred)
                task_query_losses.append(self.loss_func(query_pred, query_y))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(support_x, fast_weights, reuse=True), support_y)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if args.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.train_lr * gradients[key] for key in
                                             fast_weights.keys()]))

                    query_pred = self.forward(query_x, fast_weights, reuse=True)
                    task_query_preds.append(query_pred)
                    task_query_losses.append(self.loss_func(query_pred, query_y))

                task_output = [task_support_pred, task_query_preds, task_support_loss, task_query_losses]

                if self.classification:
                    task_support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_support_pred), 1),
                                                                   tf.argmax(support_y, 1))
                    for j in range(num_updates):
                        task_query_accs.append(
                            tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_query_preds[j]), 1),
                                                        tf.argmax(query_y, 1)))
                    task_output.extend([task_support_acc, task_query_accs])

                return task_output

            if args.norm is not 'None':
                unused = task_metalearn((self.support_x[0], self.query_x[0], self.support_y[0], self.query_y[0]),
                                        False)

            out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32] * num_updates])

            result = tf.map_fn(task_metalearn, elems=(self.support_x, self.support_y, self.query_x, self.query_y),
                               dtype=out_dtype,
                               parallel_iterations=args.meta_batch_size)
            if self.classification:
                support_preds, query_preds, support_losses, query_losses, support_accs, query_accs = result
            else:
                support_preds, query_preds, support_losses, query_losses = result

        # if 'train' in prefix:
        # self.total_loss1 = total_loss1 = tf.reduce_sum(support_losses) / tf.to_float(args.meta_batch_size)
        # self.total_losses2 = total_losses2 = [tf.reduce_mean(query_losses[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]
        self.total_loss1 = total_loss1 = tf.reduce_sum(support_losses) / tf.to_float(args.meta_batch_size)
        self.total_losses2 = total_losses2 = [tf.reduce_mean(query_losses[j]) / tf.to_float(args.meta_batch_size)
                                              for j
                                              in range(num_updates)]
        self.support_preds, self.query_preds = support_preds, query_preds
        if self.classification:
            self.total_acc1 = total_acc1 = tf.reduce_sum(support_accs) / tf.to_float(args.meta_batch_size)
            self.total_accs2 = total_accs2 = [tf.reduce_sum(query_accs[j]) / tf.to_float(args.meta_batch_size) for j
                                              in range(num_updates)]
        self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

        if args.metatrain_iterations > 0:
            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[args.num_updates - 1])
            self.metatrain_op = optimizer.apply_gradients(gvs)
        # else:
        #     self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(support_losses) / tf.to_float(args.meta_batch_size)
        #     self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(query_losses[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]
        #     if self.classification:
        #         self.metaval_total_acc1 = total_acc1 = tf.reduce_sum(support_accs) / tf.to_float(args.meta_batch_size)
        #         self.metaval_total_accs2 = total_accs2 = [tf.reduce_sum(query_accs[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]
        prefix = 'train_'
        tf.summary.scalar(prefix + 'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix + 'Pre-update accuracy', total_acc1)

        for j in range(num_updates):
            tf.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix + 'Post-update accuracy, step ' + str(j + 1), total_accs2[j])




def main():
    df = factor_history_csv()
    asset_df = df[['mom', 'beme', 'gpa', 'kospi']]
    macro_df = df[['mkt_rf', 'smb', 'hml', 'rmw', 'wml', 'call_rate', 'usdkrw']]

    N = 20
    envs_list = list(asset_df.index)
    sampler = EnvSampler(envs_list, batch_size=32)

    policy = MyPolicy()
    baseline = MyBaseline()

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau)

    sess = tf.InteractiveSession()


    for i, date_ in enumerate(envs_list):
        if i % 20 == 0:
            base_d = date_
            add_trajectory()

        envs = sampler.sample_envs(base_d, N)
        for env_t in envs:
            support_data =

















class DataGenerator(object):
    def __init__(self, config={}):
        # self.num_samples_per_class = num_samples_per_class
        self.num_classes = 2
        self.rebal_counter = 0
        self.universe = list()

        if args.datasource == 'momentum':
            self.idx_list = config.get('idx_list', args.idx_list)
            self.dim_input = config.get('num_timesteps', args.num_timesteps)
            self.dim_output = 2

        self._get_data_from_db()

    def reset_counter(self):
        self.rebal_counter = 0

    def make_data_tensor(self, base_d, train=True, test_set=False, normalize=True):

        if train:
            batch_size = args.meta_batch_size
            start_d = self.dates[max(self.dates.index(base_d) - 120, 0)]
            end_d = self.dates[self.dates.index(base_d) - 36]
            where_data = (self.data_df.date_0 >= start_d) & (self.data_df.date_0 <= end_d)

            if (self.rebal_counter == 0) or (datetime.datetime.strptime(base_d, '%Y-%m-%d').month == 12):
                self.universe = list()
                for idx in self.idx_list:
                    if (idx.lower() != 'kiscompbondcall') and len(self.data_df[(self.data_df.idx_cd == idx) & where_data]) >= 24:
                        self.universe.append(idx)
                self.rebal_counter = self.rebal_counter + 1
                print("date: {} , universe:{}".format(base_d, self.universe))
            # dataset = self.data_df[self.data_df.idx_cd.isin(self.universe) & where_data].sample(batch_size, random_state=1234)
            dataset_unbalanced = self.data_df[self.data_df.idx_cd.isin(self.universe) & where_data]

            # positive / negative data balancing
            is_positive_y = (dataset_unbalanced.t1 - dataset_unbalanced.t0 >= 0)
            dataset_support = pd.concat([dataset_unbalanced[is_positive_y].sample(batch_size // 2, random_state=1234),
                                 dataset_unbalanced[~is_positive_y].sample(batch_size // 2, random_state=1234)],
                                ignore_index=True).sample(batch_size, random_state=1234).reset_index(drop=True)
            dataset_query = pd.concat([dataset_unbalanced[is_positive_y].sample(batch_size // 2, random_state=1234),
                                 dataset_unbalanced[~is_positive_y].sample(batch_size // 2, random_state=1234)],
                                ignore_index=True).sample(batch_size, random_state=1234).reset_index(drop=True)
            dataset = pd.concat([dataset_support, dataset_query], ignore_index=True)
            # dataset = pd.concat([dataset_unbalanced[is_positive_y].sample(int(batch_size / 0.8 / 2), random_state=1234),
            #                      dataset_unbalanced[~is_positive_y].sample(int(batch_size / 0.8 / 2), random_state=1234)],
            #                     ignore_index=True).sample(int(batch_size / 0.8 / 2) * 2, random_state=1234).reset_index(drop=True)
        else:
            batch_size = 1
            if test_set:
                start_d = self.dates[self.dates.index(base_d) - 11]
                end_d = base_d
                where_data = (self.data_df.date_0 >= start_d) & (self.data_df.date_0 <= end_d)
            else:
                start_d = self.dates[self.dates.index(base_d) - 35]
                end_d = self.dates[self.dates.index(base_d) - 12]
                where_data = (self.data_df.date_0 >= start_d) & (self.data_df.date_0 <= end_d)

            dataset = self.data_df[self.data_df.idx_cd.isin(self.universe) & where_data]
        # 첫번째 또는 12월말 기준 모델 재학습

        data_tensor = np.array(dataset[self.columns_data[:-1]])
        label_tensor = np.array(dataset[self.columns_data[-1]] - dataset[self.columns_data[-2]]).reshape([-1, 1])
        label_tensor = np.concatenate((np.array(label_tensor >= 0) * 1.0, np.array(label_tensor < 0) * 1.0), axis=1)
        if normalize:
            data_tensor = (data_tensor - np.mean(data_tensor, axis=1, keepdims=True)) \
                          / np.std(data_tensor, axis=1, ddof=1, keepdims=True)

        data_tensor = data_tensor.reshape([-1, 1, args.num_timesteps])  # task_metalearn을 위한 map_fn 문제 해결 위해 1-D 추가
        label_tensor = label_tensor.reshape([-1, 1, args.num_classes])
        return data_tensor, label_tensor

    def _get_data_from_db(self):
        df = pd.read_csv('logp_modified.csv')

        timesteps = self.dim_input
        df_pivoted = df.pivot(index='eval_d', columns='idx_cd', values='log_p')
        self.dates = list(df_pivoted.index)
        self.columns_data = ['t{}'.format(i - timesteps + 1) for i in range(timesteps + 1)]
        self.data_df = pd.DataFrame(columns=['idx_cd', 'date_0'] + self.columns_data)

        for idx in self.idx_list:
            df_idx = df_pivoted[idx]
            for k in range(timesteps, len(self.dates)):
                added_row = [i for i in df_idx[(k - timesteps): (k + 1)]]
                if np.sum(np.isnan(added_row)) == 0:
                    self.data_df = self.data_df.append(
                        pd.DataFrame([[idx, df_idx.index[k - 1]] + added_row],
                                     columns=list(self.data_df.columns)), ignore_index=True)


def get_accuracy(bool_arr, bool_arr_true):
    acc = np.sum(bool_arr == bool_arr_true) / len(bool_arr)
    precision = np.sum(bool_arr & bool_arr_true) / np.sum(bool_arr_true)
    recall = np.sum(bool_arr & bool_arr_true) / np.sum(bool_arr == bool_arr_true)

    return acc, precision, recall


def normalize(inputs, activation, reuse, scope):
    if args.norm == 'batch_norm':
        return tf_layers.batch_norm(inputs, activation_fn=activation, reuse=reuse, scope=scope)
    elif args.norm == 'layer_norm':
        return tf_layers.layer_norm(inputs, activation_fn=activation, reuse=reuse, scope=scope)
    elif args.norm == 'None':
        if activation is not None:
            return activation(inputs)
        else:
            return inputs


def xent(pred, label):
    # pred = tf.reshape(pred, [-1, 1])
    # label = tf.reshape(label, [-1, 1])
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label) / args.update_batch_size


def mse(pred, label):
    pred = tf.reshape(pred, [-1, 1])
    label = tf.reshape(label, [-1, 1])
    return tf.reduce_mean(tf.square(pred-label))


# # Define our custom loss function
# def focal_loss(y_true, y_pred):
#     gamma = 2.0
#     alpha = 0.25
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#     return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def get_result(df):
    # mu = np.average(df[['port_y', 'port_bm_y', 'bm_all_y']], axis=0) * 12
    # sigma = np.std(np.array(df[['port_y', 'port_bm_y', 'bm_all_y']].values), axis=0, ddof=1) * np.sqrt(12)
    # adj_return = mu / sigma
    #
    # print("\t\t\t 'port_y', 'port_bm_y', 'bm_all_y'\nmean\t\t: {}\nsigma\t\t: {}\nadj_return\t: {}\n".format(
    #     mu, sigma, adj_return))

    x_time = pd.to_datetime(df.date_)
    plt.plot(x_time, np.array(np.cumprod(1 + df.port_y) / (1+df.port_y[0])), label='port_y')
    plt.plot(x_time, np.array(np.cumprod(1 + df.port_bm_y) / (1 + df.port_bm_y[0])), label='port_bm_y')
    plt.plot(x_time, np.array(np.cumprod(1 + df.bm_all_y) / (1 + df.bm_all_y[0])), label='bm_all_y')
    plt.legend()
    plt.yscale('log')




class MAML:
    def __init__(self, dim_input, dim_output, meta_lr=1e-3, train_lr=1e-2, test_num_updates=5):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.meta_lr = tf.placeholder_with_default(meta_lr, (), name='meta_lr')
        self.train_lr = train_lr
        self.test_num_updates = test_num_updates

        if args.datasource == 'momentum':
            self.loss_func = xent
            self.classification = True
            self.dim_hidden = [64, 32, 16]
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        else:
            raise ValueError('Unrecognized data source.')

    def construct_model(self):
        self.support_x = tf.placeholder(tf.float32, shape=[None, 1, self.dim_input], name='support_x')
        self.support_y = tf.placeholder(tf.float32, shape=[None, 1, self.dim_output], name='support_y')
        self.query_x = tf.placeholder(tf.float32, shape=[None, 1, self.dim_input], name='query_x')
        self.query_y = tf.placeholder(tf.float32, shape=[None, 1, self.dim_output], name='query_y')

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                print("1")
                training_scope.reuse_variables()
                weights = self.weights
            else:
                print("2")
                self.weights = weights = self.construct_weights()

            num_updates = self.test_num_updates
            def task_metalearn(input_data, reuse=True):
                support_x, support_y, query_x, query_y = input_data
                task_query_preds, task_query_losses = [], []

                if self.classification:
                    task_query_accs = []

                task_support_pred = self.forward(support_x, weights, reuse=reuse)
                task_support_loss = self.loss_func(task_support_pred, support_y)

                grads = tf.gradients(task_support_loss, list(weights.values()))
                if args.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]

                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(),
                                        [weights[key] - self.train_lr * gradients[key] for key in weights.keys()]))

                query_pred = self.forward(query_x, fast_weights, reuse=True)
                task_query_preds.append(query_pred)
                task_query_losses.append(self.loss_func(query_pred, query_y))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(support_x, fast_weights, reuse=True), support_y)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if args.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.train_lr * gradients[key] for key in fast_weights.keys()]))

                    query_pred = self.forward(query_x, fast_weights, reuse=True)
                    task_query_preds.append(query_pred)
                    task_query_losses.append(self.loss_func(query_pred, query_y))

                task_output = [task_support_pred, task_query_preds, task_support_loss, task_query_losses]

                if self.classification:
                    task_support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_support_pred), 1),
                                                                   tf.argmax(support_y, 1))
                    for j in range(num_updates):
                        task_query_accs.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_query_preds[j]), 1),
                                                                           tf.argmax(query_y, 1)))
                    task_output.extend([task_support_acc, task_query_accs])

                return task_output

            if args.norm is not 'None':
                unused = task_metalearn((self.support_x[0], self.query_x[0], self.support_y[0], self.query_y[0]), False)

            out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32] * num_updates])

            result = tf.map_fn(task_metalearn, elems=(self.support_x, self.support_y, self.query_x, self.query_y),
                               dtype=out_dtype,
                               parallel_iterations=args.meta_batch_size)
            if self.classification:
                support_preds, query_preds, support_losses, query_losses, support_accs, query_accs = result
            else:
                support_preds, query_preds, support_losses, query_losses = result

        # if 'train' in prefix:
        # self.total_loss1 = total_loss1 = tf.reduce_sum(support_losses) / tf.to_float(args.meta_batch_size)
        # self.total_losses2 = total_losses2 = [tf.reduce_mean(query_losses[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]
        self.total_loss1 = total_loss1 = tf.reduce_sum(support_losses) / tf.to_float(args.meta_batch_size)
        self.total_losses2 = total_losses2 = [tf.reduce_mean(query_losses[j]) / tf.to_float(args.meta_batch_size) for j
                                              in range(num_updates)]
        self.support_preds, self.query_preds = support_preds, query_preds
        if self.classification:
            self.total_acc1 = total_acc1 = tf.reduce_sum(support_accs) / tf.to_float(args.meta_batch_size)
            self.total_accs2 = total_accs2 = [tf.reduce_sum(query_accs[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]
        self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

        if args.metatrain_iterations > 0:
            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[args.num_updates - 1])
            self.metatrain_op = optimizer.apply_gradients(gvs)
        # else:
        #     self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(support_losses) / tf.to_float(args.meta_batch_size)
        #     self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(query_losses[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]
        #     if self.classification:
        #         self.metaval_total_acc1 = total_acc1 = tf.reduce_sum(support_accs) / tf.to_float(args.meta_batch_size)
        #         self.metaval_total_accs2 = total_accs2 = [tf.reduce_sum(query_accs[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]
        prefix = 'train_'
        tf.summary.scalar(prefix + 'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_acc1)

        for j in range(num_updates):
            tf.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix + 'Post-update accuracy, step ' + str(j+1), total_accs2[j])

    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, input_data, weights, reuse=False):
        input_data = tf.reshape(input_data, [-1, self.dim_input])
        hidden = normalize(tf.matmul(input_data, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i+1)]) + weights['b' + str(i+1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        out = tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights['b' + str(len(self.dim_hidden) + 1)]
        return out


def fn_train(model, saver, sess, base_d, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if args.datasource == 'momentum':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

    if args.log:
        train_writer = tf.summary.FileWriter(args.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes
    multitask_weights, reg_weights = [], []

    data_tensor, label_tensor = data_generator.make_data_tensor(base_d)
    feed_dict = {model.support_x: data_tensor[:int(len(data_tensor) * 0.6)],
                 model.support_y: label_tensor[:int(len(label_tensor) * 0.6)],
                 model.query_x: data_tensor[int(len(data_tensor) * 0.6):],
                 model.query_y: label_tensor[int(len(label_tensor) * 0.6):]}

    for itr in range(resume_itr, args.pretrain_iterations + args.metatrain_iterations):
        # print(itr)
        if itr < args.pretrain_iterations:
            ops = [model.pretrain_op]
        else:
            ops = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            # print("{} : 1".format(itr))
            ops.extend([model.summ_op, model.total_loss1, model.total_losses2[args.num_updates - 1]])
            if model.classification:
                ops.extend([model.total_acc1, model.total_accs2[args.num_updates - 1]])

        result = sess.run(ops, feed_dict=feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            # print("{} : 2".format(itr))
            prelosses.append(result[-2])
            if args.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            # print("{} : 3".format(itr))
            if itr < args.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - args.pretrain_iterations)
            print_str += ": " + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr != 0) and itr % SAVE_INTERVAL == 0:
            # print("{} : 4".format(itr))
            saver.save(sess, args.logdir + '/' + exp_string + '/model' + str(itr))

        if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
            # print("{} : 5".format(itr))
            data_tensor, label_tensor = data_generator.make_data_tensor(base_d, train=False)
            feed_dict = {model.support_x: data_tensor[:int(len(data_tensor) * 0.6)],
                         model.support_y: label_tensor[:int(len(label_tensor) * 0.6)],
                         model.query_x: data_tensor[int(len(data_tensor) * 0.6):],
                         model.query_y: label_tensor[int(len(label_tensor) * 0.6):]}

            if model.classification:
                ops = [model.total_acc1, model.total_accs2[args.num_updates - 1], model.summ_op]
            else:
                ops = [model.total_loss1, model.total_losses2[args.num_updates - 1], model.summ_op]

            result = sess.run(ops, feed_dict=feed_dict)
            print('Validation result: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, args.logdir + '/' + exp_string + '/model' + str(itr))

NUM_TEST_POINTS = 600

def fn_test(model, saver, sess, base_d, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes

    np.random.seed(1234)
    random.seed(1234)

    metaval_accs = []

    for _ in range(NUM_TEST_POINTS):
        data_tensor, label_tensor = data_generator.make_data_tensor(base_d, train=False, test_set=True)
        feed_dict = {model.support_x: data_tensor[:-1],
                     model.support_y: label_tensor[:-1],
                     model.query_x: data_tensor[-1:],
                     model.query_y: label_tensor[-1:],
                     model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([model.total_acc1] + model.total_accs2, feed_dict)
        else:
            result = sess.run([model.total_loss1] + model.total_losses2, feed_dict)
        metaval_accs.append(result)

    metaval_accs = np.array(metaval_accs)
    means = np.mean(metaval_accs, 0)
    stds = np.std(metaval_accs, 0)
    ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = args.logdir + '/' + exp_string + '/' + 'test_ubs' + str(args.update_batch_size) + '_stepsize' + str(args.update_lr) + '.csv'
    out_pkl = args.logdir + '/' + exp_string + '/' + 'test_ubs' + str(args.update_batch_size) + '_stepsize' + str(args.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accs}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update' + str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)


def mata_main():
    if args.datasource == 'momentum':
        if args.train:
            test_num_updates = 5
        else:
            test_num_updates = 10

    if args.train is False:
        orig_meta_batch_size = args.meta_batch_size
        args.meta_batch_size = 1

    if args.datasource == 'momentum':
        data_generator = DataGenerator(args.update_batch_size * 2, args.meta_batch_size)
    else:
        data_generator = DataGenerator(args.update_batch_size * 2, args.meta_batch_size)

    dim_input = data_generator.dim_input
    dim_output = data_generator.dim_output

    if args.datasource == 'momentum':
        tf_data_load = True
        num_classes = data_generator.num_classes

        if args.train:
            random.seed(1234)
            data_tensor, label_tensor = data_generator.make_data_tensor()
            support_x = tf.slice(data_tensor, [0, 0, 0], [-1, num_classes * args.update_batch_size, -1])
            query_x = tf.slice(data_tensor, [0, num_classes * args.update_batch_size, 0], [-1, -1, -1])
            support_y = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * args.update_batch_size, -1])
            query_y = tf.slice(label_tensor, [0, num_classes * args.update_batch_size, 0], [-1, -1, -1])
            input_tensors = {'support_x': support_x, 'query_x': query_x, 'support_y': support_y, 'query_y': query_y}

        random.seed(12345)
        data_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        support_x = tf.slice(data_tensor, [0, 0, 0], [-1, num_classes * args.update_batch_size, -1])
        query_x = tf.slice(data_tensor, [0, num_classes * args.update_batch_size, 0], [-1, -1, -1])
        support_y = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * args.update_batch_size, -1])
        query_y = tf.slice(label_tensor, [0, num_classes * args.update_batch_size, 0], [-1, -1, -1])
        metaval_input_tensors = {'support_x': support_x, 'query_x': query_x, 'support_y': support_y, 'query_y': query_y}
    else:
        tf_data_load = False
        input_tensors = None


    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if args.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    sess = tf.InteractiveSession()

    if args.train is False:
        args.meta_batch_size = orig_meta_batch_size

    if args.train_update_batch_size == -1:
        args.train_update_batch_size = args.update_batch_size
    if args.train_update_lr == -1:
        args.train_update_lr = args.update_lr

    exp_string = 'cls_' + str(args.num_classes) + '.mbs_' + str(args.meta_batch_size) + \
                 '.ubs_' + str(args.train_update_batch_size) + '.numstep' + str(args.num_updates) + \
                 '.updatelr' + str(args.train_update_lr)


    if args.num_filters != 64:
        exp_string += 'hidden' + str(args.num_filters)
    if args.max_pool:
        exp_string += 'maxpool'
    if args.stop_grad:
        exp_string += 'stopgrad'
    if args.baseline:
        exp_string += args.baseline
    if args.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif args.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif args.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if args.resume or not args.train:
        model_file = tf.train.latest_checkpoint(args.logdir + '/' + exp_string)
        if args.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(args.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print('Restoring model weights from ' + model_file)
            saver.restore(sess, model_file)

    if args.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)


def get_exp_string():
    exp_string = 'cls_' + str(args.num_classes) + '.mbs_' + str(args.meta_batch_size) + \
                 '.ubs_' + str(args.train_update_batch_size) + '.numstep' + str(args.num_updates) + \
                 '.updatelr' + str(args.train_update_lr)

    if args.num_filters != 64:
        exp_string += 'hidden' + str(args.num_filters)
    if args.max_pool:
        exp_string += 'maxpool'
    if args.stop_grad:
        exp_string += 'stopgrad'
    if args.baseline:
        exp_string += args.baseline
    if args.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif args.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif args.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')
    return exp_string


is_train=True
def main(is_train=True):
    # data processing
    data_generator = DataGenerator()

    if args.datasource == 'momentum':
        tf_data_load = True
        num_classes = data_generator.num_classes
        num_dates = len(data_generator.dates)
        idx_list = args.idx_list
    else:
        tf_data_load = False
        input_tensors = None
        num_dates = None

    # training and backtesting
    columns_for_store = ['date_', 'port_y', 'port_bm_y', 'bm_all_y', 'bond_y', 'n_risk', 'bm_n_risk', 'bond_wgt', 'acc', 'bm_acc', 'precision', 'bm_precision', 'recall', 'bm_recall']
    portfolio_df = pd.DataFrame(columns=columns_for_store)
    constituent_df = pd.DataFrame(columns=['date_', 'port'] + idx_list)

    dim_input = data_generator.dim_input
    dim_output = data_generator.dim_output
    model = MAML(dim_input, dim_output, test_num_updates=args.num_updates)
    model.construct_model()

    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=5)

    sess = tf.InteractiveSession()


    if args.train_update_batch_size == -1:
        args.train_update_batch_size = args.update_batch_size
    if args.train_update_lr == -1:
        args.train_update_lr = args.update_lr

    exp_string = get_exp_string()

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    # tf.train.start_queue_runners()

    if args.resume or not args.train:
        model_file = tf.train.latest_checkpoint(args.logdir + '/' + exp_string)
        if args.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(args.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print('Restoring model weights from ' + model_file)
            saver.restore(sess, model_file)


    for t in range(120, num_dates - 1):
        base_d = data_generator.dates[t]

        train = True
        fn_train(model, saver, sess, base_d, exp_string, data_generator, resume_itr)

        train = False
        fn_test(model, saver, sess, base_d, exp_string, data_generator, test_num_updates=5)
        if train is False:
            args.meta_batch_size = orig_meta_batch_size






        # bond_y = data_df[(data_df.idx_cd == 'KISCOMPBONDCALL') & (data_df.date_0 == test_d)].t1.values[0]
        bond_y = 0
        print('{}. {} AI Portfolio Result'.format(i, test_d))
        univ_long = list(test_df_logp[model.predict(test_data) >= 0.5].idx_cd)
        if len(univ_long) == 0:
            port_y = bond_y
        else:
            port_y = np.average(test_df_logp[model.predict(test_data) >= 0.5].t1) * len(univ_long) / len(universe) \
                     + bond_y * (1 - len(univ_long) / len(universe))
        acc, precision, recall = get_accuracy((model.predict(test_data) >= 0.5).reshape([-1]),
                                              np.array(test_df_logp.t1 >= 0))

        print('{}. {} BM Portfolio Result'.format(i, test_d))
        univ_bm_long = list(test_df_logp[np.prod(1 + test_data_y, axis=1) >= 1].idx_cd)
        if len(univ_bm_long) == 0:
            port_bm_y = bond_y
        else:
            port_bm_y = np.average(test_df[np.prod(1 + test_data_y, axis=1) >= 1].t1) * len(univ_bm_long) / len(universe) \
                     + bond_y * (1 - len(univ_bm_long) / len(universe))
        bm_acc, bm_precision, bm_recall = get_accuracy((np.prod(1 + test_data_y, axis=1) >= 1).reshape([-1]),
                                              np.array(test_df.t1 >= 0))

        bm_all_y = np.average(test_df.t1)

        print('{}. {} appended to the result'.format(i, test_d))
        portfolio_df = portfolio_df.append(pd.DataFrame([[test_d, port_y, port_bm_y,
                                                          bm_all_y, bond_y,
                                                          len(univ_long), len(univ_bm_long),
                                                          (1 - len(univ_long) / len(universe)), acc, bm_acc,
                                                          precision, bm_precision, recall, bm_recall]],
                                                        columns=columns_for_store), ignore_index=True)

        const_arr = np.zeros(len(idx_list))
        const_arr[list(map(idx_list.index, univ_long))] = 1
        constituent_df = constituent_df.append(
            pd.DataFrame([[test_d, 'strat'] + list(const_arr)], columns=['date_', 'port'] + idx_list),
            ignore_index=True)

        const_arr = np.zeros(len(idx_list))
        const_arr[list(map(idx_list.index, univ_bm_long))] = 1
        constituent_df = constituent_df.append(
            pd.DataFrame([[test_d, 'bm'] + list(const_arr)], columns=['date_', 'port'] + idx_list),
            ignore_index=True)

    print('plot..')
    get_result(portfolio_df)
    mill_df = portfolio_df[portfolio_df.date_ >= '2000-01-01'].reset_index()
    get_result(mill_df)

    model.save('cardiff_model.h5')

    print('save..')
    portfolio_df.to_csv("cardiff_lstm_outsample_tsmom_kospi200.csv")
    # portfolio_df.to_csv("cardiff_lstm_outsample_tsmom_mtl_balanced.csv")
    # constituent_df.to_csv("cardiff_lstm_outsample_tsmom_mtl_balanced_constituent.csv")
    print('successfully stored.')





# # # # # # # # # # # # # # # # # # # # # #  TEST  # # # # # # # # # # # # # # # #

def test():
    meta_lr = 1e-3; train_lr = 1e-2; test_num_updates = 5
    meta_lr = tf.placeholder_with_default(meta_lr, (), name='meta_lr')
    loss_func = xent
    classification = True
    dim_hidden = [16, 8]

    data_generator = DataGenerator()
    dim_input = data_generator.dim_input
    dim_output = data_generator.dim_output
    def construct_fc_weights():
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([dim_input, dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([dim_hidden[0]]))
        for i in range(1, len(dim_hidden)):
            weights['w' + str(i + 1)] = tf.Variable(
                tf.truncated_normal([dim_hidden[i - 1], dim_hidden[i]], stddev=0.01))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([dim_hidden[i]]))
        weights['w' + str(len(dim_hidden) + 1)] = tf.Variable(
            tf.truncated_normal([dim_hidden[-1], dim_output], stddev=0.01))
        weights['b' + str(len(dim_hidden) + 1)] = tf.Variable(tf.zeros([dim_output]))
        return weights

    def forward_fc(input_data, weights, reuse=False):
        input_data = tf.reshape(input_data, [-1, dim_input])
        hidden = normalize(tf.matmul(input_data, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse,
                           scope='0')
        for i in range(1, len(dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i + 1))
        return tf.matmul(hidden, weights['w' + str(len(dim_hidden) + 1)]) + weights[
            'b' + str(len(dim_hidden) + 1)]

    forward = forward_fc
    construct_weights = construct_fc_weights

    # map_fn에서 row 하나씩 slice되어 들어가므로 1-D 데이터의 경우 [1]에서 1로 바껴서 acc계산시 오류발생. shape에 1 추가함으로 일단 해결
    support_x = tf.placeholder(tf.float32, shape=[None, 1, dim_input], name='support_x')
    support_y = tf.placeholder(tf.float32, shape=[None, 1, dim_output], name='support_y')
    query_x = tf.placeholder(tf.float32, shape=[None, 1, dim_input], name='query_x')
    query_y = tf.placeholder(tf.float32, shape=[None, 1, dim_output], name='query_y')
    weights = construct_weights()
    num_updates = 5


    def task_metalearn(input_data, reuse=True):
        support_x, support_y, query_x, query_y = input_data
        task_query_preds, task_query_losses = [], []

        if classification:
            task_query_accs = []

        task_support_pred = forward(support_x, weights, reuse=reuse)
        task_support_loss = loss_func(task_support_pred, support_y)

        grads = tf.gradients(task_support_loss, list(weights.values()))
        if args.stop_grad:
            grads = [tf.stop_gradient(grad) for grad in grads]

        gradients = dict(zip(weights.keys(), grads))
        fast_weights = dict(zip(weights.keys(),
                                [weights[key] - train_lr * gradients[key] for key in weights.keys()]))

        query_pred = forward(query_x, fast_weights, reuse=True)
        task_query_preds.append(query_pred)
        task_query_losses.append(loss_func(query_pred, query_y))

        for j in range(num_updates - 1):
            loss = loss_func(forward(support_x, fast_weights, reuse=True), support_y)
            grads = tf.gradients(loss, list(fast_weights.values()))
            if args.stop_grad:
                grads = [tf.stop_gradient(grad) for grad in grads]
            gradients = dict(zip(fast_weights.keys(), grads))
            fast_weights = dict(zip(fast_weights.keys(),
                                    [fast_weights[key] - train_lr * gradients[key] for key in fast_weights.keys()]))

            query_pred = forward(query_x, fast_weights, reuse=True)
            task_query_preds.append(query_pred)
            task_query_losses.append(loss_func(query_pred, query_y))

        task_output = [task_support_pred, task_query_preds, task_support_loss, task_query_losses]

        if classification:
            task_support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_support_pred), 1),
                                                           tf.argmax(support_y, 1))

            for j in range(num_updates):
                task_query_accs.append(tf.contrib.metrics.accuracy(
                    tf.argmax(tf.nn.softmax(task_query_preds[j]), 1), tf.argmax(query_y, 1)))
            task_output.extend([task_support_acc, task_query_accs])

        return task_output

    out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]

    if classification:
        out_dtype.extend([tf.float32, [tf.float32] * num_updates])

    result = tf.map_fn(task_metalearn, elems=(support_x, support_y, query_x, query_y),
                       dtype=out_dtype,
                       parallel_iterations=args.meta_batch_size)

    if classification:
        support_preds, query_preds, support_losses, query_losses, support_accs, query_accs = result
    else:
        support_preds, query_preds, support_losses, query_losses = result

    total_loss1 = tf.reduce_sum(support_losses) / tf.to_float(args.meta_batch_size)
    total_losses2 = [tf.reduce_mean(query_losses[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]

    if classification:
        total_acc1 = tf.reduce_sum(support_accs) / tf.to_float(args.meta_batch_size)
        total_accs2 = [tf.reduce_sum(query_accs[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]
    pretrain_op = tf.train.AdamOptimizer(meta_lr).minimize(total_loss1)

    if args.metatrain_iterations > 0:
        optimizer = tf.train.AdamOptimizer(meta_lr)
        gvs = optimizer.compute_gradients(total_losses2[args.num_updates - 1])
        metatrain_op = optimizer.apply_gradients(gvs)



    sess = tf.InteractiveSession()
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    t = 120
    base_d = data_generator.dates[t]

    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if args.datasource == 'momentum':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes
    multitask_weights, reg_weights = [], []
    resume_itr = 0

    data_tensor, label_tensor = data_generator.make_data_tensor(base_d)

    data_tensor = data_tensor.reshape([-1, 1, args.num_timesteps])
    label_tensor = label_tensor.reshape([-1, 1, args.num_classes])
    feed_dict = {support_x: data_tensor[:int(len(data_tensor) * 0.5)],
                 support_y: label_tensor[:int(len(label_tensor) * 0.5)],
                 query_x: data_tensor[int(len(data_tensor) * 0.5):],
                 query_y: label_tensor[int(len(label_tensor) * 0.5):]}
    itr = 0
    for itr in range(resume_itr, args.pretrain_iterations + args.metatrain_iterations):
        ops = [metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            if classification:
                ops.extend([total_acc1, total_accs2[args.num_updates - 1]])
            ops.extend([total_loss1, total_losses2[args.num_updates - 1]])

        result_ops = sess.run(ops, feed_dict=feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result_ops[-2])
            postlosses.append(result_ops[-1])

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            if itr < args.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - args.pretrain_iterations)
            print_str += ": " + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
            data_tensor, label_tensor = data_generator.make_data_tensor(base_d, train=False)
            feed_dict = {support_x: data_tensor[:int(len(data_tensor) * 0.5)],
                         support_y: label_tensor[:int(len(label_tensor) * 0.5)],
                         query_x: data_tensor[int(len(data_tensor) * 0.5):],
                         query_y: label_tensor[int(len(label_tensor) * 0.5):]}

            if classification:
                ops = [total_acc1, total_accs2[args.num_updates - 1]]
            else:
                ops = [total_loss1, total_losses2[args.num_updates - 1]]

            result_ops = sess.run(ops, feed_dict=feed_dict)
            print('Validation result: ' + str(result_ops[0]) + ', ' + str(result_ops[1]))

