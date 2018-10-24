import numpy as np
import math
import keras
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Input, concatenate, Lambda, Conv2D, LSTM, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K


class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size):
        self.sess = sess
        # self.s_dim = state_dim
        # self.a_dim = action_dim
        self.tau = tau
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        K.set_session(sess)

        self.model, self.network_params, self.state = self.create_actor_network(state_dim, action_dim)
        self.target_model, self.target_network_params, self.target_state = self.create_actor_network(state_dim, action_dim)

        self.action_gradient = tf.placeholder(tf.float32, shape=[None, action_dim[0]])
        # self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        self.unnormalized_actor_grad = tf.gradients(self.model.output, self.network_params, -self.action_gradient)
        self.actor_grad = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_grad))
        grads = zip(self.actor_grad, self.network_params)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
        self.sess.run(tf.global_variables_initializer())

    def create_actor_network(self, state_dim, action_dim):
        raise NotImplementedError

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads,})

    def target_train(self):
        actor_weights = self.model.trainable_weights
        actor_target_weights = self.target_model.trainable_weights
        for i in range(len(actor_weights)):
            self.sess.run(actor_target_weights[i].assign(
                tf.multiply(actor_weights[i], self.tau) + tf.multiply(actor_target_weights[i], 1-self.tau)))

    def get_num_trainable_vars(self):
        return self.num_trainable_vars




