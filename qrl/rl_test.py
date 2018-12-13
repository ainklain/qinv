
import numpy as np
import pandas as pd
import gym
from gym import spaces

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Conv2D, BatchNormalization, Reshape, LeakyReLU
from keras.optimizers import Adam
from keras import backend as K

from qtest.nalu import NALU
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

# from qrl.env_portfolio2 import PortfolioEnv
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


def main():
    history, stock_list, marketdate, item_list = read_stock_history()
    history = history[:, :, :4]
    target_stocks = stock_list
    num_training_time = 2000
    window_length = 50
    nb_actions = len(target_stocks) + 1

    # action_dim = [nb_actions]
    # state_dim = [window_length, nb_actions]
    # batch_size = 64
    # action_bound = 1.
    # tau = 1e-3
    # learning_rate = 1e-4
    #
    # predictor_type = 'cnn'
    # use_batch_norm = True
    #
    # CONFIG = {'seed': 1234,
    #           'episode': 1,
    #           'batch_size': 256,
    #           'gamma': 0.99,
    #           'buffer_size': 100,
    #           'max_step': 500,
    #           'tau': 0.001
    #           }

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
    # x = NALU(nb_actions)(x)
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
    l1 = Activation('relu')(x)
    x = Conv2D(32, kernel_size=(1, 1))(l1)
    x = BatchNormalization()(x)
    l2 = Activation('relu')(x)
    flattened_observation = Flatten()(l2)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(64)(x)
    l3 = Activation('relu')(x)
    x = Dense(64)(l3)
    l4 = Activation('relu')(x)
    w_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
    x = Dense(1, activation='linear', kernel_initializer=w_init)(l4)
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
    agent_history = agent.fit(env, nb_steps=1000000, visualize=False, verbose=1, nb_max_episode_steps=252)



    # After training is done, we save the final weights.
    agent.save_weights('ddpg_{}_weights.h5f'.format('RL_ENV2_TEST'), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=252)


def main2():

    history, factor_id, marketdate = factor_history()
    target_assets = factor_id
    window_length = 50
    mem_size = 100
    steps = 252
    nb_actions = len(target_assets)

    # get target history
    import copy
    target_history = copy.deepcopy(history)
    target_marketdate = copy.deepcopy(marketdate)

    env = PortfolioEnv(target_history, target_assets, target_marketdate, steps=steps, window_length=window_length,
                       trading_cost=0.001)

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1, window_length, nb_actions, 1), name='observation_input')
    reshaped_obs_input = Reshape((window_length, nb_actions, 1))(observation_input)
    x = Conv2D(10, kernel_size=(30, 1))(reshaped_obs_input)
    x = BatchNormalization()(x)
    x = Activation('linear', name='actor_layer_1')(x)
    x = LeakyReLU()(x)
    x = Conv2D(10, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('linear', name='actor_layer_2')(x)
    x = LeakyReLU()(x)
    flattened_observation = Flatten()(x)
    x = Dense(64, activation='linear')(flattened_observation)
    x = LeakyReLU()(x)
    x = Dense(32, activation='linear')(x)
    x = LeakyReLU(name='actor_layer_3')(x)
    w_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
    x = Dense(nb_actions, activation='softmax', kernel_initializer=w_init)(x)
    actor = Model(inputs=observation_input, outputs=x)
    actor_intermediate_1 = Model(inputs=actor.inputs, outputs=actor.get_layer('actor_layer_1').output)
    actor_intermediate_2 = Model(inputs=actor.inputs, outputs=actor.get_layer('actor_layer_2').output)
    actor_intermediate_3 = Model(inputs=actor.inputs, outputs=actor.get_layer('actor_layer_3').output)
    print(actor.summary())

    # x = Conv2D(32, kernel_size=(50, 1))(reshaped_obs_input)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2D(32, kernel_size=(1, 1))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # flattened_observation = Flatten()(x)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(64, activation='linear', name='critic_layer_1')(x)
    x = LeakyReLU()(x)
    x = Dense(32, activation='linear', name='critic_layer_2')(x)
    x = LeakyReLU()(x)
    w_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
    x = Dense(1, activation='linear', kernel_initializer=w_init, name='critic_layer_3')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    critic_intermediate_1 = Model(inputs=critic.inputs, outputs=critic.get_layer('critic_layer_1').output)
    critic_intermediate_2 = Model(inputs=critic.inputs, outputs=critic.get_layer('critic_layer_2').output)
    critic_intermediate_3 = Model(inputs=critic.inputs, outputs=critic.get_layer('critic_layer_3').output)
    print(critic.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=mem_size, nb_steps_warmup_actor=mem_size,
                      random_process=random_process, gamma=.90, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.

    n_loop = 10
    output_actor_1 = np.zeros([n_loop, actor_intermediate_1.output_shape[1], actor_intermediate_1.output_shape[2]])
    output_actor_2 = np.zeros([n_loop, actor_intermediate_2.output_shape[1], actor_intermediate_2.output_shape[2]])
    output_actor_3 = np.zeros([n_loop, actor_intermediate_3.output_shape[1]])
    output_critic_1 = np.zeros([n_loop, critic_intermediate_1.output_shape[1]])
    output_critic_2 = np.zeros([n_loop, critic_intermediate_2.output_shape[1]])
    output_critic_3 = np.zeros([n_loop, critic_intermediate_3.output_shape[1]])

    for l in range(n_loop):
        agent_history = agent.fit(env, nb_steps=steps*50, visualize=False, verbose=1, nb_max_episode_steps=steps)
        # obs, _, _ = env.src._step()
        recent_obs = agent.recent_observation.reshape(1, 1, window_length, nb_actions, 1)
        recent_action = agent.recent_action.reshape(1, nb_actions)
        output_actor_1[l] = actor_intermediate_1.predict(recent_obs)[0, :, :, 0]
        output_actor_2[l] = actor_intermediate_2.predict(recent_obs)[0, :, :, 0]
        output_actor_3[l] = actor_intermediate_3.predict(recent_obs)
        # print("1: {}, 2:{}, 3:{}, 0's:{}, 1's:{}".format(np.mean(output_actor_1.squeeze(), axis=2),
        #                                                  np.mean(output_actor_2.squeeze(), axis=2),
        #                                                  output_actor_3,
        #                                                  np.sum(output_actor_3 == 0),
        #                                                  np.sum(output_actor_3 > 0)))

        output_critic_1[l] = critic_intermediate_1.predict([recent_action, recent_obs])
        output_critic_2[l] = critic_intermediate_2.predict([recent_action, recent_obs])
        output_critic_3[l] = critic_intermediate_3.predict([recent_action, recent_obs])
        print("1: {}, 2:{}, 3:{}".format(output_critic_1[l], output_critic_2[l], output_critic_3[l]))

        agent.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=steps)
    plot_layer_3d(output_actor_1)
    plot_layer_3d(output_actor_2)
    plot_layer_2d(output_actor_3)
    plot_layer_2d(output_critic_1)
    plot_layer_2d(output_critic_2)
    plot_layer_2d(output_critic_3)

    # test
    asset_list = target_assets
    trading_cost = 0.0
    time_cost = 0.00
    window_length = 50
    sample_start_date = None
    num_assets = len(asset_list)
    src = DataGenerator(history, asset_list, marketdate,
                        steps=steps,
                        window_len=window_length,
                        start_date=sample_start_date)
    sim = PortfolioSim(asset_names=asset_list,
                       steps=steps,
                       trading_cost=trading_cost,
                       time_cost=time_cost)


def plot_layer_2d(mat):

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x, y = np.meshgrid(range(mat.shape[1]), range(mat.shape[0]))
    xs = x.reshape(-1)
    ys = y.reshape(-1)
    cols = mat.reshape(-1) - np.min(mat)
    colors = cm.hsv(cols / max(cols))


    colormap = cm.ScalarMappable(cmap=cm.hsv)
    colormap.set_array(cols)
    ax.scatter(xs, ys, c=colors)
    cb = fig.colorbar(colormap)
    plt.title("subtracted:{}".format(np.min(mat)))
    plt.show()



def plot_layer_3d(mat):

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = np.meshgrid(range(mat.shape[1]),
                          range(mat.shape[0]),
                          range(mat.shape[2]))
    xs = x.reshape(-1)
    ys = y.reshape(-1)
    zs = z.reshape(-1)
    cols = mat.reshape(-1) - np.min(mat)
    colors = cm.hsv(cols / max(cols))


    colormap = cm.ScalarMappable(cmap=cm.hsv)
    colormap.set_array(cols)
    ax.scatter(xs, ys, zs, c=colors)
    cb = fig.colorbar(colormap)
    plt.title("subtracted:{}".format(np.min(mat)))
    plt.show()



if __name__ == '__main__':
    main2()


#
# https://www.alexirpan.com/2018/02/14/rl-hard.html
# https://himanshusahni.github.io/2018/02/23/reinforcement-learning-never-worked.html
