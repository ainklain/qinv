from qdeep.base.base_model import BaseModel
import tensorflow as tf

class FundaModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.init_saver()

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


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)




