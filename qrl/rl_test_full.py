
import os
import random
import numpy as np
import pandas as pd
from collections import deque
import tensorflow as tf

from qrl.ou import OrnsteinUhlenbeck
import gym
from gym import spaces

from qrl.env_factor import PortfolioEnv


def read_stock_history():
    from qdata.dbmanager import SqlManager
    sql_ = """
    select p.infocode, marketdate, open_ * a.CumAdjFactor as open_, high * a.CumAdjFactor as high_, low * a.CumAdjFactor as low_, close_ * a.CumAdjFactor as close_, volume / a.CumAdjFactor as volume_
	from qai..ds2primqtprc P
	join (
		select a.infocode, a.AdjDate, isnull(a.EndAdjDate, '9999-12-31') as endadjdate, a.CumAdjFactor
			from qai..ds2adj A
			where a.infocode in (46238,46712,36100,39988,46171,46113,40142,39985,46244,52853,51959,63649,46669,72983,50565,66846,52588,46415)
			and a.adjtype = 2
	) A
	on p.infocode = a.infocode
	and p.marketdate >= a.AdjDate and p.MarketDate <= a.EndAdjDate
	where p.infocode in (46238,46712,36100,39988,46171,46113,40142,39985,46244,52853,51959,63649,46669,72983,50565,66846,52588,46415)
	and p.marketdate >= '2000-01-01'
	order by p.infocode, MarketDate
    """

    sqlm = SqlManager()
    df = sqlm.db_read(sql_)

    item_list = ['open_', 'high_', 'low_', 'close_', 'volume_']
    infocode = list(df.infocode.unique())
    marketdate = list(df.marketdate.unique())

    df_to_arr = np.zeros([len(df.marketdate.unique()), len(df.infocode.unique()), len(item_list)])
    for i, item in enumerate(item_list):
        df_to_arr[:, :, i] = pd.pivot_table(df, index='marketdate', columns='infocode', values=item)

    history = df_to_arr[~np.isnan(np.sum(df_to_arr, axis=(1, 2))), :, :]

    return history, infocode, marketdate, item_list


def factor_history_csv():
    file_nm = 'data_factor.csv'
    df = pd.read_csv(file_nm, index_col=0)

    df.columns = [i.lower() for i in df.columns]
    df.set_index('eval_d', inplace=True)
    df = df[df.isna().sum(axis=1) == 0]
    factor_id = list(df.columns)
    marketdate = list(df.index.unique())
    history = df.values

    return history, factor_id, marketdate


def factor_history():
    from qdata.dbmanager import SqlManager
    sql_ = """
        select * 
            from  (
            select idxcd, d.eval_d, round(clsprc / lag(clsprc, 1) over (partition by idxcd order by base_d) - 1, 5) as value_
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
            order by eval_d
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

    return history, factor_id, marketdate


class Memory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory_counter = 0
        self.memory = deque()

    def add(self, s, a, r, s_, done):
        experience = (s, a, r, s_, done)
        if self.memory_counter < self.memory_size:
            self.memory.append(experience)
            self.memory_counter += 1
        else:
            self.memory.popleft()
            self.memory.append(experience)

    def clear(self):
        self.memory.clear()
        self.memory_counter = 0

    def sample_batch(self, batch_size):
        if self.memory_counter < batch_size:
            print('insufficient memory')
            return random.sample(self.memory, self.memory_counter)
            # return False
        else:
            return random.sample(self.memory, batch_size)


class DDPG(object):
    def __init__(self, sess, a_dim, s_dim, a_bound, memory, gamma, tau, lr_a, lr_c, batch_size, ):
        self.sess = sess
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.s_dim = s_dim
        self.memory = memory
        self.batch_size = batch_size

        self.S = tf.placeholder(tf.float32, [None] + s_dim, 's')
        self.S_ = tf.placeholder(tf.float32, [None] + s_dim, 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Shared'):
            o_shared = self._build_shared_net(self.S, scope='eval', trainable=True)
            o_shared_ = self._build_shared_net(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Actor'):
            self.a = self._build_a(o_shared, scope='eval', trainable=True)
            a_ = self._build_a(o_shared_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            q = self._build_c(o_shared, self.a, scope='eval', trainable=True)
            q_ = self._build_c(o_shared_, a_, scope='target', trainable=False)

        self.se_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Shared/eval')
        self.st_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Shared/target')
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.soft_replace = [tf.assign(t, (1-tau) * t + tau * e)
                             for t, e in zip(self.at_params + self.ct_params + self.st_params,
                                             self.ae_params + self.ce_params + self.se_params)]

        q_target = self.R + gamma * q_

        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(lr_c).minimize(td_error, var_list=self.ce_params)

        a_loss = -tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(lr_a).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        self.sess.run(self.soft_replace)

        mini_batch = self.memory.sample_batch(self.batch_size)
        if not mini_batch:
            return False

        s, a, r, s_, done = [np.array([i_row[j] for i_row in mini_batch])
                             for j in range(5)]

        self.sess.run(self.atrain, {self.S: s})
        self.sess.run(self.ctrain, {self.S: s, self.a: a, self.R: r, self.S_: s_})

    def store_transition(self, s, a, r, s_, done):
        self.memory.add(s, a, r, s_, done)

    def _build_shared_net(self, s, scope, trainable):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            s_reshaped = tf.reshape(s, [-1] + self.s_dim + [1])
            x = tf.layers.conv2d(s_reshaped, filters=10, kernel_size=(30, 1), trainable=trainable)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=10, kernel_size=(1, 1), trainable=trainable)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            flattened_obs = tf.layers.flatten(x)

            return flattened_obs

    def _build_a(self, o_shared, scope, trainable):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(o_shared, 64, activation=tf.nn.relu, trainable=trainable)
            x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=trainable)
            a = tf.layers.dense(x, self.a_dim, activation=tf.nn.softmax,
                                kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003, seed=None),
                                trainable=trainable)

            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, o_shared, a, scope, trainable):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [o_shared.shape.as_list()[1], n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(o_shared, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 1, trainable=trainable)
            return net


def main():

    history, factor_id, marketdate = factor_history_csv()
    target_assets = factor_id
    window_length = 50
    mem_size = 100
    steps = 252
    trading_cost = -0.001

    gamma = 0.9
    tau = 0.01
    lr_a = 0.001
    lr_c = 0.001

    batch_size = 100

    nb_actions = len(target_assets)

    max_episodes = 1
    max_ep_steps = 250
    RENDER = True
    LOAD = True

    # get target history
    import copy
    target_history = copy.deepcopy(history)
    target_marketdate = copy.deepcopy(marketdate)

    env = PortfolioEnv(target_history, target_assets, target_marketdate, steps=steps, window_length=window_length,
                       trading_cost=trading_cost)

    a_dim = env.action_space.shape[0]
    s_dim = list(env.observation_space.shape)
    a_bound = env.action_space.high
    # register session
    sess = tf.Session()
    # K.set_session(sess)
    action_noise = OrnsteinUhlenbeck(mu=np.zeros(a_dim))
    memory = Memory(mem_size)
    agent = DDPG(sess, a_dim, s_dim, a_bound=a_bound, memory=memory, gamma=gamma, tau=tau, lr_a=lr_a, lr_c=lr_c, batch_size=batch_size)


    #Saver
    SAVER_DIR = ""
    saver = tf.train.Saver(max_to_keep=5)
    checkpoint_path = os.path.join(SAVER_DIR, "model")
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

    if LOAD:
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    else:
        sess.run(tf.global_variables_initializer())

    for i in range(max_episodes):
        s = env.reset()
        ep_reward = 1
        for j in range(max_ep_steps):
            if RENDER:
                env.render()

            a = agent.choose_action(s) + action_noise() / 10.
            np.clip(a, 0., 0.9999)
            a[-1] = 1 - np.sum(a[:-1])  # 남는건 채권

            s_, r, done, info = env.step(a)

            agent.store_transition(s, a, [r], s_, done)

            agent.learn()

            s = s_
            ep_reward *= (1. + r)

            # print(a)

            if j == max_ep_steps - 1:
                # import matplotlib.animation as animation
                # import matplotlib.pyplot as plt
                # plt.axis('off')
                # fig = plt.figure()
                # ani = animation.ArtistAnimation(fig, env.ims)
                # ani.save('test.gif', writer='imagemagick', dpi=80)
                print('episode: {}, reward: {}'.format(i, ep_reward))
                break

        if i % 20 == 0:
            ckpt_path = os.path.join(checkpoint_path, 'DDPG_mcost.ckpt')
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)
            save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
            print('\nSave Model %s\n' % save_path)

    sess.close()
#
# if __name__ == '__main__':
#     main()

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimage
# from matplotlib import animation
#
# fig = plt.figure()
#
# plt.axis('off')
# a = []
# num = [i * 20 for i in range(1, 3)]
# for i in num:
#     im_data = mpimage.imread('fig_{}.jpg'.format(i))
#     im = plt.imshow(im_data, animated=True)
#     a.append([im])
#
# my_anim = animation.ArtistAnimation(fig, a, interval=1000, blit=True, repeat_delay=1000)
# my_anim.save('test.mp4', fps=30)
# plt.show()
#
# https://www.alexirpan.com/2018/02/14/rl-hard.html
# https://himanshusahni.github.io/2018/02/23/reinforcement-learning-never-worked.html
