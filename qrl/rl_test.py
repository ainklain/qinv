
import numpy as np
import pandas as pd
import gym
from gym import spaces

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Conv2D, BatchNormalization, Reshape
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from qrl.env_portfolio2 import PortfolioEnv


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





def main():
    history, stock_list, marketdate, item_list = read_stock_history()
    history = history[:, :, :4]
    target_stocks = stock_list
    num_training_time = 2000
    window_length = 50
    nb_actions = len(target_stocks) + 1

    action_dim = [nb_actions]
    state_dim = [window_length, nb_actions]
    batch_size = 64
    action_bound = 1.
    tau = 1e-3
    learning_rate = 1e-4

    predictor_type = 'cnn'
    use_batch_norm = True

    CONFIG = {'seed': 1234,
              'episode': 1,
              'batch_size': 256,
              'gamma': 0.99,
              'buffer_size': 100,
              'max_step': 500,
              'tau': 0.001
              }

    # get target history
    target_history = np.empty(shape=(num_training_time, len(target_stocks), history.shape[2]))
    target_marketdate = marketdate[:num_training_time]

    for i, stock in enumerate(target_stocks):
        target_history[:, i, :] = history[:num_training_time, stock_list.index(stock), :]



    env = PortfolioEnv(target_history, target_stocks, target_marketdate, steps=252, window_len=window_length)


    np.random.seed(123)
    env.seed(123)

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1, window_length, nb_actions, 1), name='observation_input')
    reshaped_obs_input = Reshape((window_length, nb_actions, 1))(observation_input)
    x = Conv2D(32, kernel_size=(3, 1))(reshaped_obs_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    w_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
    x = Dense(nb_actions, activation='softmax', kernel_initializer=w_init)(x)
    actor = Model(inputs=observation_input, outputs=x)

    # actor = Sequential()
    # actor.add(Conv2D(32, kernel_size=(1, 3), input_shape=state_dim + [1]))
    # actor.add(BatchNormalization())
    # actor.add(Activation('relu'))
    # actor.add(Conv2D(32, kernel_size=(1, 1)))
    # actor.add(BatchNormalization())
    # actor.add(Activation('relu'))
    # actor.add(Flatten())
    # actor.add(Dense(64, activation='relu'))
    # actor.add(Dense(64, activation='relu'))
    # w_init = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
    # actor.add(Dense(nb_actions, activation='softmax', kernel_initializer=w_init))
    print(actor.summary())

    x = Conv2D(32, kernel_size=(3, 1))(reshaped_obs_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    flattened_observation = Flatten()(x)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    w_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
    x = Dense(1, activation='linear', kernel_initializer=w_init)(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    agent.fit(env, nb_steps=500, visualize=True, verbose=1, nb_max_episode_steps=252)


    # After training is done, we save the final weights.
    # agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    # agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)


if __name__ == '__main__':
    main()
