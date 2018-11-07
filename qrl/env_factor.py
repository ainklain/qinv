# https://github.com/hackthemarket/gym-trading/blob/master/gym_trading/envs/trading_env.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint
import gym
import gym.spaces
from gym.utils import seeding


import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class DataGenerator:
    def __init__(self, history, asset_list, marketdate, steps=730, window_len=50, start_date=None):
        import copy

        self.step = 0
        self.steps = steps + 1
        self.window_len = window_len
        self.start_date = start_date

        self._data = history.copy()
        self.asset_names = copy.copy(asset_list)
        self.marketdate = copy.copy(marketdate)

    def _step(self):
        self.step += 1
        obs = self.data[self.step: (self.step + self.window_len), :].copy()

        ground_truth_obs = self.data[(self.step + self.window_len): (self.step + self.window_len) + 1, :].copy()
        done = self.step >= self.steps
        return obs, done, ground_truth_obs

    def reset(self):
        self.step = 0

        if self.start_date is None:
            self.idx = np.random.randint(
                low=self.window_len,
                high=self._data.shape[0] - self.steps)
        else:
            self.idx = self.marketdate.index(np.datetime64(self.start_date))

        data = self._data[(self.idx - self.window_len): (self.idx + self.steps + 1), :]
        self.data = data

        return self.data[self.step: (self.step + self.window_len), :].copy(), \
               self.data[(self.step + self.window_len): (self.step + self.window_len + 1), :].copy()


class PortfolioSim(object):
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, asset_names=list(), steps=250, trading_cost=1e-3, time_cost=1e-4):
        self.asset_names = asset_names

        # invariant for object life
        self.trading_cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        # change every step
        self.step = 0
        self.actions = np.zeros([self.steps, len(self.asset_names)])
        self.navs = np.ones(self.steps)
        self.stk_nav = np.ones([self.steps, len(self.asset_names)])
        self.strat_returns = np.ones(self.steps)
        self.positions = np.zeros([self.steps, len(self.asset_names)])
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros([self.steps, len(self.asset_names)])
        self.stk_returns = np.ones([self.steps, len(self.asset_names)])

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.stk_nav.fill(1)
        self.strat_returns.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.stk_returns.fill(0)

    def _step(self, actions, stk_returns):
        """ actions: weights
            stk_returns: y of stocks
        """
        eps = 1e-8

        if self.step == 0:
            last_pos = np.zeros(len(actions))
            last_nav = 1.
            stk_nav = np.ones(len(actions))
        else:
            last_pos = self.positions[self.step - 1, :]
            last_nav = self.navs[self.step - 1]
            stk_nav = self.stk_nav[self.step - 1, :]

        self.stk_returns[self.step, :] = stk_returns
        self.actions[self.step, :] = actions

        self.positions[self.step, :] = ((stk_returns + 1.) * actions) / (np.dot((stk_returns + 1.), actions) + eps)
        self.trades[self.step, :] = actions - last_pos

        trade_costs_pct = np.sum(abs(self.trades[self.step, :])) * self.trading_cost
        self.costs[self.step] = trade_costs_pct + self.time_cost
        reward = (np.dot((stk_returns + 1.), actions) - 1.) - self.costs[self.step]
        self.strat_returns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = last_nav * (1. + self.strat_returns[self.step])
            self.stk_nav[self.step, :] = stk_nav * (1. + self.stk_returns[self.step, :])

        info = {'reward': reward, 'nav': self.navs[self.step], 'costs': self.costs[self.step]}

        done = (self.navs[self.step] == 0)

        self.step += 1
        return reward, info, done

    def to_df(self):
        """returns internal state in new dataframe """
        cols = ['action', 'bod_nav', 'mkt_nav', 'mkt_return', 'sim_return',
                'position', 'costs', 'trade']
        # pdb.set_trace()
        df = pd.DataFrame({'action': self.actions,  # today's action (from agent)
                           'bod_nav': self.navs,  # BOD Net Asset Value (NAV)
                           'mkt_nav': self.mkt_nav,
                           'mkt_return': self.mkt_retrns,
                           'sim_return': self.strat_retrns,
                           'position': self.posns,  # EOD position
                           'costs': self.costs,  # eod costs
                           'trade': self.trades},  # eod trade
                          columns=cols)
        return df


class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, history, asset_list, marketdate, steps=730, trading_cost=0.0,  # 0.0025
                 time_cost=0.00,
                 window_length=50, sample_start_date=None):
        assert history.shape[0] == len(marketdate)
        assert history.shape[1] == len(asset_list)

        super().__init__()
        self.window_length = window_length
        self.num_assets = len(asset_list)
        self.asset_list = asset_list
        self.steps = steps

        self.src = DataGenerator(history, asset_list, marketdate,
                                 steps=steps,
                                 window_len=window_length,
                                 start_date=sample_start_date)

        self.sim = PortfolioSim(asset_names=asset_list,
                                steps=steps,
                                trading_cost=trading_cost,
                                time_cost=time_cost)

        self.action_space = gym.spaces.Box(0, 1, shape=(self.num_assets, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(window_length, self.num_assets),
                                        dtype=np.float32)

    def step(self, action):
        return self._step(action)

    def _step(self, action, eps=1e-8):
        np.testing.assert_almost_equal(action.shape, (self.num_assets,))
        # action = np.clip(action, 0, 1)
        #
        # weights = action
        # weights /= (weights.sum() + eps)
        # weights[0] += np.clip(1 - weights.sum(), 0, 1)
        action = np.clip(action, 0, 1)
        weights = np.zeros_like(action)
        weights[action > 0] = action[action > 0] / (np.sum(action[action > 0]) + eps)
        # weights[action < 0] = action[action < 0] / (np.sum(action[action < 0]) + eps)


        assert (weights >= (-1 - 1e-6) * (weights <= (1 + 1e-6))).all()
        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3)
        # np.testing.assert_almost_equal(np.sum(weights), 0.0, 3)

        obs, done1, ground_truth_obs = self.src._step()

        y1 = obs[-1, :]
        reward, info, done2 = self.sim._step(weights, y1)

        self.infos.append(info)

        obs = np.expand_dims(obs, -1)
        # print('reward:{} , info:{}'.format(reward, info))
        return obs, reward, done1 or done2, info

    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        obs, ground_truth_obs = self.src.reset()
        obs = np.expand_dims(obs, -1)

        self.render_call = 0

        return obs
        # return obs, info

    def render(self, mode='human', close=False):
        return self._render(mode=mode, close=close)

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            if self.render_call == 0:
                self.fig = plt.figure()
                self.ax1, self.ax2, self.ax3 = self.fig.subplots(3, 1)

            self._get_image()

            if self.render_call == 0:
                self.ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fancybox=True, shadow=True)
                self.render_call += 1

            # self.fig.canvas.draw()
            if self.sim.step % 20 == 0:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    def _get_image(self):
        df_info = pd.DataFrame(self.infos)
        self.ax1.plot(df_info.index, df_info.nav, color='k')

        actions_t = np.transpose(self.sim.actions)
        pal = sns.color_palette("hls", self.num_assets)
        self.ax2.stackplot(df_info.index, actions_t[:, :len(df_info.index)], colors=pal)

        stk_nav_t = np.transpose(self.sim.stk_nav)
        for i in range(self.num_assets):
            self.ax3.plot(df_info.index, stk_nav_t[i, :len(df_info.index)], color=pal[i], label=self.asset_list[i])

