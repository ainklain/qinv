import numpy as np
import math
import keras
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Activation
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf


class CriticNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau):
        self.sess = sess
        self.action_dim = action_dim
        self.tau = tau
        self.learning_rate = learning_rate

        K.set_session(sess)

        self.model, self.action, self.state = self.create_critic_network(state_dim, action_dim)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_dim, action_dim)

        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.global_variables_initializer())

    def create_critic_network(self, state_dim, action_dim):
        raise NotImplementedError

    def target_train(self):
        critic_weights = self.model.trainable_weights
        critic_target_weights = self.target_model.trainable_weights
        for i in range(len(critic_weights)):
            self.sess.run(critic_target_weights[i].assign(
                tf.multiply(critic_weights[i], self.tau) + tf.multiply(critic_target_weights[i], 1-self.tau)))

    def action_gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions})


