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


def _prices2returns(prices):
    px = pd.DataFrame(prices)
    nl = px.shift().fillna(0)
    R = ((px - nl)/nl).fillna(0).replace([np.inf, -np.inf], np.nan).dropna()
    R = np.append(R[0].values, 0)

    return R


class DataGenerator:
    def __init__(self, history, stock_list, marketdate, steps=730, window_len=50, start_date=None):
        import copy

        self.steps = steps + 1
        self.window_len = window_len
        self.start_date = start_date

        self._data = history.copy()
        self.asset_names = copy.copy(stock_list)
        self.marketdate = copy.copy(marketdate)

    def _step(self):
        self.step += 1
        obs = self.data[self.step: (self.step + self.window_len), :, :].copy()

        ground_truth_obs = self.data[(self.step + self.window_len): (self.step + self.window_len) + 1, :, :].copy()
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

        data = self._data[(self.idx - self.window_len): (self.idx + self.steps + 1), :, :4]
        self.data = data

        return self.data[self.step: (self.step + self.window_len), :, :].copy(), \
               self.data[(self.step + self.window_len): (self.step + self.window_len + 1), :, :].copy()


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
        self.strat_returns.fill(1)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.stk_returns.fill(1)

    def _step(self, actions, stk_returns):
        """ actions: weights
            stk_returns: y of stocks
        """
        assert stk_returns[0] == 1.0
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

        self.positions[self.step, :] = (stk_returns * actions) / (np.dot(stk_returns, actions) + eps)
        self.trades[self.step, :] = actions - last_pos

        trade_costs_pct = np.sum(abs(self.trades[self.step, :])) * self.trading_cost
        self.costs[self.step] = trade_costs_pct + self.time_cost
        reward = np.dot(stk_returns, actions) - self.costs[self.step]
        self.strat_returns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = last_nav * self.strat_returns[self.step]
            self.stk_nav[self.step, :] = stk_nav * self.stk_returns[self.step, :]

        info = {'reward': reward, 'nav': self.navs[self.step], 'costs': self.costs[self.step]}

        done = (self.navs[self.step] == 0)

        self.step += 1
        return reward, info, done

    def to_df(self):
        """returns internal state in new dataframe """
        cols = ['action', 'bod_nav', 'mkt_nav', 'mkt_return', 'sim_return',
                'position', 'costs', 'trade']
        rets = _prices2returns(self.navs)
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

    def __init__(self, history, stock_list, marketdate, steps=730, trading_cost=0.0025, time_cost=0.00,
                 window_len=50, sample_start_date=None):
        assert history.shape[0] == len(marketdate)
        assert history.shape[1] == len(stock_list)

        super().__init__()
        self.window_len = window_len
        self.num_stocks = len(stock_list)
        self.viewer = None
        self.steps = steps

        self.src = DataGenerator(history, stock_list, marketdate,
                                 steps=steps,
                                 window_len=window_len,
                                 start_date=sample_start_date)

        asset_names = ['cash'] + stock_list
        self.sim = PortfolioSim(asset_names=asset_names,
                                steps=steps,
                                trading_cost=trading_cost,
                                time_cost=time_cost)

        self.action_space = gym.spaces.Box(0, 1, shape=(self.num_stocks + 1, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(window_len, self.num_stocks, history.shape[-1]),
                                        dtype=np.float32)

    def step(self, action):
        return self._step(action)

    def _step(self, action, eps=1e-8):
        np.testing.assert_almost_equal(action.shape, (self.num_stocks + 1,))
        action = np.clip(action, 0, 1)

        weights = action
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)

        assert ((action >= 0) * (action <= 1)).all()
        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3)

        obs, done1, ground_truth_obs = self.src._step()

        cash_obs = np.ones((self.window_len, 1, obs.shape[2]))
        obs = np.concatenate((cash_obs, obs), axis=1)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=1)

        close_p_vector = obs[-1, :, 3]
        open_p_vector = obs[-1, :, 0]
        y1 = close_p_vector / open_p_vector
        reward, info, done2 = self.sim._step(weights, y1)

        self.infos.append(info)
        obs = obs[:, :, 3:4] / obs[:, :, 0:1]
        return obs, reward, done1 or done2, info


    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        obs, ground_truth_obs = self.src.reset()
        cash_obs = np.ones((self.window_len, 1, obs.shape[2]))
        obs = np.concatenate((cash_obs, obs), axis=1)
        # cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        # ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=1)

        self.fig = plt.figure()
        self.ax1, self.ax2, self.ax3 = self.fig.subplots(3, 1)
        # ax2 = self.fig.add_subplot(21)

        # ax = self.fig.add_subplot(111)
        # self.line, = ax.plot([], [], 'k-')


        obs = obs[:, :, 3:4] / obs[:, :, 0:1]
        return obs
        # return obs, info

    def render(self, mode='human', close=False):
        return self._render(mode=mode, close=close)

    def _render(self, mode='human', close=False):
        self._get_image()
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            # if self.fig is None:
            #     self.fig = plt.figure()
            #
            #     ax = self.fig.add_subplot(111)
            #     self.line, = ax.plot([], [])

            # self.plot()

            # self.fig.canvas.draw()
            if self.sim.step % 20 == 0:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    def plot(self):
        x = np.arange(self.steps)
        y = np.zeros_like(x)

        for i, info in enumerate(self.infos):
            y[i] = info['nav']
        self.line.set_data(x, y)
        return self.line,



    def _get_image(self):
        # return self.infos[-1]['market_value']

        import io
        from PIL import Image
        # from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        # from matplotlib.figure import Figure
        # plt.figure()
        # plt.plot([1, 2])
        # plt.title("test")
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # buf.seek(0)
        # im = Image.open(buf)
        # im.show()
        # buf.close()

        # plt.figure()
        # df_info = pd.DataFrame(self.infos)
        # plt.plot(df_info.index, df_info.nav)
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # buf.seek(0)
        # im = Image.open(buf)
        # buf.close()
        #
        # fig = Figure()
        # canvas = FigureCanvas(fig)
        # canvas = self.fig.canvas
        # ax = self.fig.gca()
        # ax1, ax2 = self.fig.subplots(2, 1)
        df_info = pd.DataFrame(self.infos)
        self.ax1.plot(df_info.index, df_info.nav, color='k')

        actions_t = np.transpose(self.sim.actions)
        pal = sns.color_palette("hls", 19)
        self.ax2.stackplot(df_info.index, actions_t[:, :len(df_info.index)], colors=pal)

        stk_nav_t = np.transpose(self.sim.stk_nav)
        for i in range(self.num_stocks + 1):
            self.ax3.plot(df_info.index, stk_nav_t[i, :len(df_info.index)], color=pal[i])

        # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        # w, h = canvas.get_width_height()
        # image.shape = (w, h, 3)
        # return image
