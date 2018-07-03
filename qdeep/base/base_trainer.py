import tensorflow as tf


class BaseTrainer:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        # self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # self.sess.run(self.init)

    def train(self):
        prev_l = 99999999999
        patience_cnt = 0
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            loss_avg = self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

            if prev_l - loss_avg > 0.005:
                patience_cnt = 0
                prev_l = loss_avg
            else:
                patience_cnt += 1

            print("[Train] Epoch: {} Finished, loss: {:.3f}, patience: {}".format(cur_epoch, loss_avg, patience_cnt))
            if patience_cnt > self.config.patience:
                print('early stopping')
                break

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
