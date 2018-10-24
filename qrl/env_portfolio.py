import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint
import gym
import gym.spaces
from gym.utils import seeding


def index_to_date(index):
    import datetime

    start_date = '2012-08-13'
    end_date = '2017-08-11'
    date_format = '%Y-%m-%d'
    start_datetime = datetime.datetime.strptime(start_date, date_format)
    end_datetime = datetime.datetime.strptime(end_date, date_format)
    return (start_datetime + datetime.timedelta(index)).strftime(date_format)



def sharpe_ratio(returns, freq=250, rfr=0, eps=1e-8):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


def max_drawdown(returns, eps=1e-8):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + eps)



class DataGenerator:
    def __init__(self, history, stock_list, steps=730, window_len=50, start_idx=0, start_date=None):
        import copy

        self.steps = steps + 1
        self.window_len = window_len
        self.start_idx = start_idx
        self.start_date = start_date

        self._data = history.copy()
        self.asset_names = copy.copy(stock_list)

    def _step(self):
        self.step += 1
        obs = self.data[:, self.step: (self.step + self.window_len), :].copy()

        ground_truth_obs = self.data[:, (self.step + self.window_len): (self.step + self.window_len) + 1, :].copy()
        done = self.step >= self.steps
        return obs, done, ground_truth_obs

    def reset(self):
        self.step = 0

        if self.start_date is None:
            self.idx = np.random.randint(
                low=self.window_len,
                high=self._data.shape[1] - self.steps)
        else:
            self.idx = index_to_date(self.start_date) - self.start_idx

        data = self._data[:, (self.idx - self.window_len): (self.idx + self.steps + 1), :4]
        self.data = data

        return self.data[:, self.step: (self.step + self.window_len), :].copy(), \
               self.data[:, (self.step + self.window_len): (self.step + self.window_len + 1), :].copy()


class PortfolioSim:
    def __init__(self, asset_names=list(), steps=730, trading_cost=0.0025, time_cost=0.0):
        self.asset_names = asset_names
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps

    def _step(self, w1, y1, eps=1e-8):
        assert w1.shape == y1.shape
        assert y1[0] == 1.0

        p0 = self.p0
        dw1 = (y1 * w1) / (np.dot(y1, w1) + eps)
        mu1 = self.cost * (np.abs(dw1 - w1)).sum()
        assert mu1 < 1.0

        p1 = p0 * (1. - mu1) * np.dot(y1, w1)
        p1 = p1 * (1. - self.time_cost)
        rho1 = p1 / p0 - 1.

        r1 = np.log((p1 + eps) / (p0 + eps))
        reward = r1 / self.steps * 1000.
        self.p0 = p1

        done = (p1 == 0)

        info = {
            'reward': reward,
            'log_return': r1,
            'portfolio_value': p1,
            'return': y1.mean(),
            'rate_of_return': rho1,
            'weights_mean': w1.mean(),
            'weights_std': w1.std(),
            'cost': mu1,
        }

        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.p0 = 1.0


class PortfolioEnv(gym.Env):

    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(self,
                 history,
                 stock_list,
                 steps=730,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_len=50,
                 start_idx=0,
                 sample_start_date=None):
        super().__init__()
        self.window_len = window_len
        self.num_stocks = history.shape[0]
        self.start_idx = start_idx

        self.src = DataGenerator(history, stock_list, steps=steps, window_len=window_len, start_idx=start_idx, start_date=sample_start_date)

        self.sim = PortfolioSim(asset_names=stock_list,
                                trading_cost=trading_cost,
                                time_cost=time_cost,
                                steps=steps)

        self.action_space = gym.spaces.Box(0, 1, shape=(len(self.src.asset_names) + 1, ), dtype=np.float32)
        self.obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(len(stock_list), window_len, history.shape[-1]),
                                        dtype=np.float32)

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self, action):
        return self._step(action)

    def _step(self, action, eps=1e-8):
        np.testing.assert_almost_equal(action.shape, (len(self.sim.asset_names) + 1, ))
        action = np.clip(action, 0, 1)

        weights = action
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)

        assert ((action >= 0) * (action <= 1)).all()
        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3)

        obs, done1, ground_truth_obs = self.src._step()

        cash_obs = np.ones((1, self.window_len, obs.shape[2]))
        obs = np.concatenate((cash_obs, obs), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        close_p_vector = obs[:, -1, 3]
        open_p_vector = obs[:, -1, 0]
        y1 = close_p_vector / open_p_vector
        reward, info, done2 = self.sim._step(weights, y1)

        info['market_value'] = np.cumprod([inf['return'] for inf in self.infos + [info]])[-1]
        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step)
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs

        self.infos.append(info)

        return obs, reward, done1 or done2, info

    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        obs, ground_truth_obs = self.src.reset()
        cash_obs = np.ones((1, self.window_len, obs.shape[2]))
        obs = np.concatenate((cash_obs, obs), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info = {}
        info['next_obs'] = ground_truth_obs
        return obs, info

    def render(self, mode='human', close=False):
        return self._render(mode=mode, close=close)

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            self.plot()

    def plot(self):
        df_info = pd.DataFrame(self.infos)
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sr = sharpe_ratio(df_info.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sr)
        df_info[['portfolio_value', 'market_value']].plot(title=title, fig=plt.gcf(), rot=30)



class MultiActionPortfolioEnv(PortfolioEnv):
    def __init__(self, history, stock_list, model_names,
                 steps=730,
                 trading_cost=0.0025,
                 time_cost=0.0,
                 window_len=50,
                 start_idx=0,
                 sample_start_date=None):
        super().__init__(history, stock_list, steps, trading_cost, time_cost, window_len, start_idx, sample_start_date)
        self.model_names = model_names
        self.sim = [PortfolioSim(asset_names=stock_list, trading_cost=trading_cost, time_cost=time_cost, steps=steps)
                    for _ in range(len(self.model_names))]

    def _step(self, action, eps=1e-8):
        assert action.ndim == 2
        assert action.shape[1] == len(self.sim[0].asset_names) + 1
        assert action.shape[0] == len(self.model_names)

        action = np.clip(action, 0, 1)
        weights = action
        weights /= (np.sum(weights, axis=1, keepdims=True) + eps)
        weights[:, 0] += np.clip(1 - np.sum(weights, axis=1), 0, 1)
        assert ((action >= 0) * (action <= 1)).all()
        np.testing.assert_almost_equal(np.sum(weights, axis=1), np.ones(shape=(weights.shape[0])), 3)
        obs, done1, ground_truth_obs = self.src._step()

        cash_obs = np.ones((1, self.window_len, obs.shape[2]))
        obs = np.concatenate((cash_obs, obs), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        close_p_vector = obs[:, -1, 3]
        open_p_vector = obs[:, -1, 0]
        y1 = close_p_vector / open_p_vector

        rewards = np.empty(shape=(weights.shape[0]))
        info = {}
        dones = np.empty(shape=(weights.shape[0]), dtype=bool)
        for i in range(weights.shape[0]):
            reward, current_info, done2 = self.sim[i]._step(weights[i], y1)
            rewards[i] = reward
            info[self.model_names[i]] = current_info['portfolio_value']
            info['return'] = current_info['return']
            dones[i] = done2

        info['market_value'] = np.cumprod([inf['return'] for inf in self.infos + [info]])[-1]
        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step)
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs

        self.infos.append(info)

        return obs, rewards, np.all(dones) or done1, info

    def _reset(self):
        self.infos = []
        for sim in self.sim:
            sim.reset()
        obs, ground_truth_obs = self.src.reset()
        cash_obs = np.ones((1, self.window_len, obs.shape[2]))
        obs = np.concatenate((cash_obs, obs), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info = {}
        info['next_obs'] = ground_truth_obs
        return obs, info

    def plot(self):
        df_info = pd.DataFrame(self.infos)
        fig = plt.gcf()
        title = 'Performance of Various Models'
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        df_info[self.model_names + ['market_value']].plot(title=title, fig=fig, rot=30)











