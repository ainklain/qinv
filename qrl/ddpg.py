

import json
import os
import numpy as np
from copy import deepcopy
from keras import backend as K
import keras.optimizers as optimizers
from qrl.replay_buffer import ReplayBuffer
from rl.core import Agent

from qrl.util import *

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


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class DDPGAgent(Agent):
    def __init__(self, nb_actions, actor, critic, actor_noise, memory,
                 obs_normalizer=None, action_processor=None,
                 # config_file='config/default.json',
                 config=None,
                 model_save_path='weights/ddpg/ddpg/ckpt',
                 summary_path='results/ddpg/', **kwargs):

        super(__class__, self).__init__(**kwargs)

        assert config is not None, 'should set config'
        self.config = config
        np.random.seed(self.config['seed'])

        self.model_save_path = model_save_path
        self.summary_path = summary_path

        # Parameters
        self.nb_actions = nb_actions

        # Related objects
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.memory = memory
        self.obs_normalizer = obs_normalizer
        self.action_processor = action_processor
        self.summary_ops, self.summary_vars = build_summaries()

        # State
        self.compiled = False
        self.reset_states()

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        critic_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                critic_inputs.append(i)
        combined_inputs[self.critic_action_input_idx] = self.actor(critic_inputs)

        combined_output = self.critic(combined_inputs)

        updates = actor_optimizer.get_updates(
            params=self.actor.trainable_weights, loss=-K.mean(combined_output))
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        self.actor_train_fn = K.function(critic_inputs + [K.learning_phase()],
                                         [self.actor(critic_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action


    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.

        # Returns
            List of metrics values
        """


    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1, visualize=False, debug=False):

        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        self.actor.target_train()
        self.critic.target_train()

        np.random.seed(self.config['seed'])
        num_episode = self.config['episode']
        batch_size = self.config['batch_size']
        gamma = self.config['gamma']
        self.buffer = ReplayBuffer(self.config['buffer_size'])

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    # callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    else:
                        observation = observation[:, :, 3:4] / observation[:, :, 0:1]
                    assert observation is not None

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

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
                        print('Episode: {}, Reward: {}'.format(i, total_reward))
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










