from qdeep.base.base_model import BaseModel
import tensorflow as tf


class FundaModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, inputs):
        self.is_training = tf.placeholder(tf.bool)

        with tf.variable_scope('fc'):
            net = tf.layers.dense(inputs, self.config.n_hidden1, tf.nn.relu)
            # net = tf.layers.dropout(net, rate=0.5, training=is_training)
            net = tf.layers.batch_normalization(net)
            net = tf.layers.dense(net,  self.config.n_hidden2, tf.nn.relu)
            # net = tf.layers.dropout(net, rate=0.5, training=is_training)
            net = tf.layers.batch_normalization(net)
            self.predicted = tf.layers.dense(net,  self.config.n_output, tf.nn.relu, name='predicted')




