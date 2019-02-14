import numpy as np
import pandas as pd
import tensorflow as tf

import keras
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense, Input, Add
from keras.models import Model, Sequential
from keras import backend as K

from qrl.actor import ActorNetwork
from qrl.critic import CriticNetwork
from qrl.env_portfolio import PortfolioEnv
from qrl.ou import OrnsteinUhlenbeck
from qrl.ddpg import DDPG



def res_block(inputs, trainable):
    x_shortcut = inputs
    net = Conv2D(32, kernel_size=(1, 1), trainable=trainable)(inputs)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = Conv2D(32, kernel_size=(1, 1), trainable=trainable)(net)
    net = BatchNormalization()(net)
    net = keras.layers.Add()([net, x_shortcut])
    net = LeakyReLU()(net)

    return net

def stock_predictor(inputs, predictor_type, use_batch_norm):
    window_len = inputs.get_shape()[2]
    assert predictor_type in ['cnn', 'lstm']
    if predictor_type == 'cnn':
        net = Conv2D(32, kernel_size=(1, 3))(inputs)
        if use_batch_norm:
            net = BatchNormalization()(net)
        net = LeakyReLU()(net)

        n_res = 1
        for _ in range(n_res):
            net = res_block(net, True)
        net = Conv2D(32, kernel_size=(1, 1))(net)
        if use_batch_norm:
            net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = Flatten()(net)

    elif predictor_type == 'lstm':
        pass
    else:
        raise NotImplementedError

    return net


class StockActor(ActorNetwork):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size, predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        super().__init__(sess, state_dim, action_dim, learning_rate, tau, batch_size)

    def create_actor_network(self, state_dim, action_dim):
        nb_classes, window_len = state_dim
        assert nb_classes == action_dim[0]
        assert window_len > 2

        inputs = Input(shape=state_dim + [1], name='input')

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)

        model_actor = Sequential()
        model_actor.add(Dense(64, activation='linear'))
        if self.use_batch_norm:
            model_actor.add(BatchNormalization())
        model_actor.add(LeakyReLU())
        model_actor.add(Dense(64, activation='linear'))
        if self.use_batch_norm:
            model_actor.add(BatchNormalization())
        model_actor.add(LeakyReLU())
        w_init = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        out = Dense(action_dim[0], activation='softmax', kernel_initializer=w_init)(model_actor(net))
        model = Model(inputs=inputs, outputs=out)

        return model, model.trainable_weights, inputs

    # def train(self, states, action_grads):
    #     self.sess.run(self.optimize, feed_dict={
    #         self.state: states,
    #         self.action_gradient: action_grads,})


class StockCritic(CriticNetwork):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        super().__init__(sess, state_dim, action_dim, learning_rate, tau)

    def create_critic_network(self, state_dim, action_dim):
        inputs = Input(shape=state_dim + [1])
        action = Input(shape=action_dim)

        net1 = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)

        net1 = Dense(64, activation='linear')(net1)
        net1 = LeakyReLU()(net1)
        net2 = Dense(64, activation='linear')(action)

        net = Add()([net1, net2])
        if self.use_batch_norm:
            net = BatchNormalization()(net)
        w_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
        out = Dense(action_dim[0], activation='softmax', kernel_initializer=w_init)(net)

        model = Model(inputs=[inputs, action], outputs=out)
        model.compile(loss='mse', optimizer='adam')
        return model, action, inputs


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

    df_to_arr = np.zeros([len(df.infocode.unique()), len(df.marketdate.unique()), len(item_list)])
    for i, item in enumerate(item_list):
        df_to_arr[:, :, i] = pd.pivot_table(df, index='infocode', columns='marketdate', values=item)

    history = df_to_arr[:, ~np.isnan(np.sum(df_to_arr, axis=(0, 2))), :]

    return history, infocode, item_list


def test_model(env, model):
    observation, info = env.reset()
    observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    done = False
    while not done:
        action = model.predict_single(observation)
        observation, _, done, _ = env.step(action)
        observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    env.render()


def main():
    history, stock_list, item_list = read_stock_history()
    history = history[:, :, :4]
    target_stocks = stock_list
    num_training_time = 1095
    window_len = 50
    nb_classes = len(target_stocks) + 1

    CONFIG = {'seed': 1234,
              'episode': 1,
              'batch_size': 256,
              'gamma': 0.99,
              'buffer_size': 100,
              'max_step': 500,
              'tau': 0.001
              }

    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[stock_list.index(stock), :num_training_time, :]


    env = PortfolioEnv(target_history, target_stocks, steps=1000, window_len=window_len)

    action_dim = [nb_classes]
    state_dim = [nb_classes, window_len]
    batch_size = 64
    action_bound = 1.
    tau = 1e-3
    learning_rate = 1e-4

    predictor_type = 'cnn'
    use_batch_norm = True

    actor_noise = OrnsteinUhlenbeck(mu=np.zeros(action_dim))
    sess = tf.Session()
    K.set_session(sess)
    actor = StockActor(sess, state_dim, action_dim, learning_rate, tau, batch_size, predictor_type, use_batch_norm)
    critic = StockCritic(sess, state_dim, action_dim, learning_rate, tau, predictor_type, use_batch_norm)
    model_ddpg = DDPG(env, sess, actor, critic, actor_noise, config=CONFIG)

    model_ddpg.initialize()
    model_ddpg.train()
