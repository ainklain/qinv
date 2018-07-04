from qdeep.base.base_trainer import BaseTrainer
import numpy as np
import tensorflow as tf
import os


class FundaTrainer(BaseTrainer):
    def __init__(self, sess, model, data, config, logger):
        super().__init__(sess, model, data, config, logger)
        self.build_trainer()

    def build_trainer(self):
        inputs, labels = self.data.next_batch
        # self.predicted = self.model.build_model(inputs)
        self.model.build_model(inputs)

        train_vars_fc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc')

        with tf.variable_scope('loss'):
            # mask = np.zeros(labels.shape[1])
            # mask[4] = 1
            # mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            # loss = tf.add(tf.losses.mean_squared_error(labels=labels, predictions=predicted),
            #               0.75 * tf.reduce_sum(tf.cast(tf.square(labels - predicted), tf.float32) * mask))
            self.loss = tf.losses.mean_squared_error(labels=labels, predictions=self.model.predicted)
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                   var_list=train_vars_fc,
                                                                                   global_step=self.model.global_step_tensor)
            tf.summary.scalar('loss', self.loss)

        tf.add_to_collection('predicted', self.model.predicted)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('optimizer', self.optimizer)

        self.merged = tf.summary.merge_all()
        # self.logger.train_summary_writer.add_graph(self.sess.graph)
        self.logger.initialize()
        # self.sess.run(tf.global_variables_initializer())
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)

    def train_epoch(self):
        self.sess.run(self.data.train_init_op)
        num_iter_per_epoch = self.data.n_train // self.config.batch_size

        loss_avg = 0
        for i in range(num_iter_per_epoch):
            l, summary = self.train_step()
            # if i % 200 == 0:
            cur_iter = self.model.global_step_tensor.eval(self.sess)
            self.logger.train_summary_writer.add_summary(summary, cur_iter * num_iter_per_epoch + i)
            loss_avg += l

        loss_avg = loss_avg / num_iter_per_epoch

        # summaries_dict = {
        #     'loss': loss_avg,
        # }
        # self.logger.summarize(cur_iter, summaries_dict=summaries_dict)
        self.model.save(self.sess)

        return loss_avg

    def train_step(self):
        _, loss, summary = self.sess.run([self.optimizer, self.loss, self.merged])
        return loss, summary

    def predict(self, x_data, is_training=True):
        if is_training is True:
            self.sess.run(self.data.test_init_op, feed_dict={x_test: x_data})

        # x_data = np.random.random([1, self.config.n_input])
        predict_value = self.sess.run([self.model.predicted])

        # https: // github.com / tensorflow / tensorflow / issues / 13865
        pass








# def funda_train(sess, model, data, config, logger=None):
#     num_iter_per_epoch = data.n_train // config.batch_size
#
#     # with tf.variable_scope('dataa', reuse=True):
#     inputs, labels = data.next_batch
#     train_init_op = data.train_init_op
#
#     model.build_model(inputs)
#
#     train_vars_fc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc')
#
#     with tf.variable_scope('loss'):
#         loss = tf.losses.softmax_cross_entropy(labels, tf.cast(model.logits, tf.float32))
#         optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(loss,
#                                                                           var_list=train_vars_fc,
#                                                                           global_step=model.global_step_tensor)
#         tf.summary.scalar('loss', loss)
#
#     # with tf.variable_scope('accuracy'):
#     #     prediction = tf.argmax(model.logits, 1)
#     #     equality = tf.equal(prediction, tf.argmax(labels, 1))
#     #     accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
#     #     tf.summary.scalar('accuracy', accuracy)
#
#     tf.add_to_collection('loss', loss)
#     tf.add_to_collection('optimizer', optimizer)
#
#     init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#     sess.run(init)
#
#     merged = tf.summary.merge_all()
#     summary_writer = tf.summary.FileWriter(os.path.join(config.summary_dir, "train"), sess.graph)
#     sess.run(train_init_op)
#     losses_ep_ae = []
#     for cur_epoch in range(model.cur_epoch_tensor.eval(sess), config.num_epochs + 1, 1):
#         losses = []
#         for _ in range(num_iter_per_epoch):
#             feed_dict = {model.is_training: True}
#             summary, _, loss = sess.run([merged, optimizer, loss], feed_dict=feed_dict)
#             summary = sess.run([merged], feed_dict={model.is_training: False})
#             summary_writer.add_summary(summary, cur_epoch)
#
#             losses.append(loss)
#
#         loss_avg = np.mean(losses)
#         losses_ep_ae.append(loss_avg)
#         print("cur_epoch: {}  loss:{}".format(cur_epoch, loss_avg))
#         sess.run(model.increment_cur_epoch_tensor)
#
#         cur_it = model.global_step_tensor.eval(sess)
#         summaries_dict = {
#             'loss': loss,
#         }
#         # logger.summarize(cur_it, summaries_dict=summaries_dict)
#         # model_ae.save(sess)
#
#         # summary_writer.add_summary(summary, cur_it)
#
#     # test_init_op = iterator.make_initializer(data.test_dataset)
#     # sess.run(test_init_op)
#     # sess.run(model_ae.encoded)
