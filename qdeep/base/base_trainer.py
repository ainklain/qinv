import tensorflow as tf
import os


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

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess,
                        os.path.join(self.config.checkpoint_dir, self.config.model_name),
                        self.model.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError