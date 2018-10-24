

import json
import os
import numpy as np
import tensorflow as tf

from qrl.replay_buffer import ReplayBuffer


CONFIG = {'seed': 1234,
          'episode': 10,
          'batch_size': 32,
          'gamma': 0.99,
          'buffer_size': 100,
          'max_step': 100,
          'tau': 0.001
          }
# {
#   "episode": 500,
#   "max step": 1000,
#   "buffer size": 100000,
#   "batch size": 64,
#   "tau": 0.001,
#   "gamma": 0.99,
#   "actor learning rate": 0.0001,
#   "critic learning rate": 0.001,
#   "seed": 1337
# }

def build_summaries():
    return None, None

class DDPG(object):
    def __init__(self, env, sess, actor, critic, actor_noise, obs_normalizer=None, action_processor=None,
                 # config_file='config/default.json',
                 config=CONFIG,
                 model_save_path='weights/ddpg/ddpg/ckpt',
                 summary_path='results/ddpg/'):
        # with open(config_file) as f:
        #     self.config = json.load(f)
        # assert self.config != None, "Can't load config file."
        self.config = config
        np.random.seed(self.config['seed'])
        if env:
            env.seed(self.config['seed'])

        self.model_save_path = model_save_path
        self.summary_path = summary_path

        self.sess = sess
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.obs_normalizer = obs_normalizer
        self.action_processor = action_processor
        self.summary_ops, self.summary_vars = build_summaries()

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def train(self, save_every_episode=1, verbose=True, debug=False, seed=1234):
        self.actor.target_train()
        self.critic.target_train()

        np.random.seed(self.config['seed'])
        num_episode = self.config['episode']
        batch_size = self.config['batch_size']
        gamma = self.config['gamma']
        self.buffer = ReplayBuffer(self.config['buffer_size'])

        for i in range(num_episode):
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer: " + str(self.buffer.count()))

            obs_t, _ = self.env.reset()
            if self.obs_normalizer:
                s_t = self.obs_normalizer(obs_t)
            else:
                s_t = obs_t[:, :, 3:4] / obs_t[:, :, 0:1]

            # ep_reward = 0
            # ep_avg_max_q = 0
            total_reward = 0
            for j in range(self.config['max_step']):
                loss = 0
                a_t = self.actor.model.predict(np.expand_dims(s_t, axis=0)).squeeze(axis=0) + self.actor_noise()

                if self.action_processor:
                    action_take = self.action_processor(a_t)
                else:
                    action_take = a_t

                obs_t1, r_t, done, info = self.env.step(action_take)

                if self.obs_normalizer:
                    s_t1 = self.obs_normalizer(obs_t1)
                else:
                    s_t1 = obs_t1[:, :, 3:4] / obs_t1[:, :, 0:1]

                self.buffer.add(s_t, a_t, r_t, s_t1, done)

                # if self.buffer.count() >= batch_size:
                # s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(batch_size)
                batch = self.buffer.sample_batch(batch_size)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[1] for e in batch])

                # y_i = []
                target_q = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])
                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + gamma * target_q[k]

                # predicted_q = self.critic.model.train_on_batch([states, actions], y_t)
                # ep_avg_max_q += np.amax(predicted_q)
                loss += self.critic.model.train_on_batch([states, actions], y_t)

                a_outs = self.actor.model.predict(states)
                grads = self.critic.action_gradients(states, a_outs)
                self.actor.train(states, grads[0])

                self.actor.target_train()
                self.critic.target_train()

                total_reward += r_t
                s_t = s_t1

                print("Episode", i, "Step", j, "Action", a_t, "Reward", r_t, "Loss", loss)
                if done or j == self.config['max_step'] - 1:
                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}'.format(i, total_reward))
                    break

        print('Finish.')

    def predict(self, obs):
        if self.obs_normalizer:
            obs = self.obs_normalizer
        action = self.actor.model.predict(obs)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def predict_single(self, obs):
        if self.obs_normalizer:
            obs = self.obs_normalizer(obs)
        action = self.actor.model.predict(np.expand_dims(obs, axis=0)).squeeze(axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def save_model(self):
        self.actor.model.save_weights("actor_model.h5", overwrite=True)
        with open('actor_model.json', 'w') as outfile:
            json.dump(self.actor.model.to_json(), outfile)

        self.critic.model.save_weights("critic_model.h5", overwrite=True)
        with open('critic_model.json', 'w') as outfile:
            json.dump(self.critic.model.to_json(), outfile)










