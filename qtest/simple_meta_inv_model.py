from qdata.dbmanager import SqlManager

import argparse
import csv
import datetime
import numpy as np
import pandas as pd
import pickle
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.layers.python import layers as tf_layers

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, BatchNormalization, Input, RepeatVector
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras import backend as K


parser = argparse.ArgumentParser()
parser.add_argument('--datasource', default='momentum')


class Argument:
    def __init__(self):
        self.datasource = 'momentum'
        self.idx_list = ['MSCIWORLD', 'MSCIEM', 'KOSPI200', 'KOSDAQ150', 'SPX500', 'NASDAQ100', 'RUSSELL2000',
                    'CSI300', 'HSCE', 'TOPIX100', 'MSCIUSREITTR', 'USDKRW', 'GSCIGOLD', 'KISCOMPBONDCALL']

        self.num_timesteps = 12
        self.stop_grad = True
        self.meta_batch_size = 25
        self.update_batch_size = 5
        self.train = True
        self.train_update_batch_size = -1
        self.train_update_lr = -1
        self.update_lr = 1e-3
        self.num_classes = 2
        self.num_updates = 1
        self.num_filters = 64
        self.max_pool = False
        self.baseline = None
        self.norm = None
        self.resume = True
        self.logdir = '/tmp/data'
        self.test_iter = -1
        self.pretrain_iterations = 0
        self.meta_batch_size = 25
        self.log = True

args = Argument()


class DataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1

        if args.datasource == 'momentum':
            self.idx_list = config.get('idx_list', args.idx_list)
            self.dim_input = config.get('num_timesteps', args.num_timesteps)
            self.dim_output = 1

        self.get_data_from_db()

    def make_data_tensor(self, universe, start_d, end_d, train=True):
        if train:
            return_df = self.data_df[self.data_df.idx_cd.isin(universe) &
                                     (self.data_df.date_0 >= start_d) & (self.data_df.date_0 < end_d)]


    def get_data_from_db(self):
        df = pd.read_csv('logp.csv')

        timesteps = self.dim_input
        df_pivoted = df.pivot(index='eval_d', columns='idx_cd', values='log_p')
        data_df = pd.DataFrame(columns=['idx_cd', 'date_0'] + ['t{}'.format(i-timesteps+1) for i in range(timesteps+1)])

        n_date = len(df_pivoted.index)
        for idx in self.idx_list:
            df_idx = df_pivoted[idx]
            for k in range(timesteps, n_date):
                added_row = [i for i in df_idx[(k - timesteps): (k + 1)]]
                if np.sum(np.isnan(added_row)) == 0:
                    data_df = data_df.append(pd.DataFrame([[idx, df_idx.index[k - 1]] + added_row],
                                                          columns=list(data_df.columns)), ignore_index=True)

        self.data_df = data_df




def get_accuracy(bool_arr, bool_arr_true):
    acc = np.sum(bool_arr == bool_arr_true) / len(bool_arr)
    precision = np.sum(bool_arr & bool_arr_true) / np.sum(bool_arr_true)
    recall = np.sum(bool_arr & bool_arr_true) / np.sum(bool_arr == bool_arr_true)

    return acc, precision, recall


def normalize(inputs, activation, reuse, scope):
    if args.norm == 'batch_norm':
        return tf_layers.batch_norm(inputs, activation_fn=activation, reuse=reuse, scope=scope)
    elif args.norm == 'layer_norm':
        return tf_layers.layer_norm(inputs, activation_fn=activation, reuse=reuse, scope=scope)
    elif args.norm == 'None':
        if activation is not None:
            return activation(inputs)
        else:
            return inputs


def xent(pred, label):
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / args.update_batch_size


def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))


# Define our custom loss function
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def get_result(df):
    # mu = np.average(df[['port_y', 'port_bm_y', 'bm_all_y']], axis=0) * 12
    # sigma = np.std(np.array(df[['port_y', 'port_bm_y', 'bm_all_y']].values), axis=0, ddof=1) * np.sqrt(12)
    # adj_return = mu / sigma
    #
    # print("\t\t\t 'port_y', 'port_bm_y', 'bm_all_y'\nmean\t\t: {}\nsigma\t\t: {}\nadj_return\t: {}\n".format(
    #     mu, sigma, adj_return))

    x_time = pd.to_datetime(df.date_)
    plt.plot(x_time, np.array(np.cumprod(1 + df.port_y) / (1+df.port_y[0])), label='port_y')
    plt.plot(x_time, np.array(np.cumprod(1 + df.port_bm_y) / (1 + df.port_bm_y[0])), label='port_bm_y')
    plt.plot(x_time, np.array(np.cumprod(1 + df.bm_all_y) / (1 + df.bm_all_y[0])), label='bm_all_y')
    plt.legend()
    plt.yscale('log')


class MAML:
    def __init__(self, dim_input, dim_output, meta_lr=1e-3, train_lr=1e-2, test_num_updates=5):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.meta_lr = meta_lr
        self.train_lr = train_lr
        self.test_num_updates = test_num_updates

        if args.datasource == 'momentum':
            self.loss_func = xent
            self.classification = True
            self.dim_hidden = [64, 32, 16]
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        else:
            raise ValueError('Unrecognized data source.')

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        if input_tensors is None:
            self.support_x = tf.placeholder(tf.float32)
            self.support_y = tf.placeholder(tf.float32)
            self.query_x = tf.placeholder(tf.float32)
            self.query_y = tf.placeholder(tf.float32)
        else:
            self.support_x = input_tensors['support_x']
            self.support_y = input_tensors['support_y']
            self.query_x = input_tensors['query_x']
            self.query_y = input_tensors['query_y']

        with tf.variable_scope('model', reuse=False) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()

            support_losses, support_preds, query_losses, query_preds = [], [], [], []
            support_accs, query_accs = [], []
            num_updates = max(self.test_num_updates, self.test_num_updates)  # max(self.test_num_updates, FLAGS.num_updates)
            query_preds = [[]] * num_updates
            query_losses = [[]] * num_updates
            query_accs = [[]] * num_updates

            def task_metalearn(input_data, reuse=True):
                support_x, support_y, query_x, query_y = input_data
                task_query_preds, task_query_losses = [], []

                if self.classification:
                    task_query_accs = []

                task_support_pred = self.forward(support_x, weights, reuse=reuse)
                task_support_loss = self.loss_func(task_support_pred, support_y)

                grads = tf.gradients(task_support_loss, list(weights.values))
                if args.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]

                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(),
                                        [weights[key] - self.update_lr * gradients[key] for key in weights.keys()]))

                query_pred = self.forward(query_x, fast_weights, reuse=True)
                task_query_preds.append(query_pred)
                task_query_losses.append(self.loss_func(query_pred, query_y))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(support_x, fast_weights, reuse=True), support_y)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if args.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.update_lr * gradients[key] for key in fast_weights.keys()]))

                    query_pred = self.forward(query_x, fast_weights, reuse=True)
                    task_query_preds.append(query_pred)
                    task_query_losses.append(self.loss_func(query_pred, query_y))

                task_output = [task_support_pred, task_query_preds, task_support_loss, task_query_losses]

                if self.classification:
                    task_support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_support_pred), 1),
                                                                   tf.argmax(support_y, 1))
                    for j in range(num_updates):
                        task_query_accs.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_query_preds[j]), 1),
                                                                           tf.argmax(query_y, 1)))
                    task_output.extend([task_support_acc, task_query_accs])

                return task_output

            if args.norm is not 'None':
                unused = task_metalearn((self.support_x[0], self.query_x[0], self.support_y[0], self.query_y[0]), False)

            out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32] * num_updates])
            result = tf.map_fn(task_metalearn, elems=(self.support_x, self.query_x, self.support_y, self.query_y),
                               dtype=out_dtype,
                               parallel_iterations=args.meta_batch_size)
            if self.classification:
                support_preds, query_preds, support_losses, query_losses, support_accs, query_accs = result
            else:
                support_preds, query_preds, support_losses, query_losses = result


        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(support_losses) / tf.to_float(args.meta_batch_size)
            self.total_losses2 = total_lossses2 = [tf.reduce_mean(query_losses[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]

            self.support_preds, self.query_preds = support_preds, query_preds
            if self.classification:
                self.total_acc1 = total_acc1 = tf.reduce_sum(support_accs) / tf.to_float(args.meta_batch_size)
                self.total_accs2 = total_accs2 = [tf.reduce_sum(query_accs[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if args.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[args.num_updates - 1])
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(support_losses) / tf.to_float(args.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(query_losses[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]
            if self.classification:
                self.metaval_total_acc1 = total_acc1 = tf.reduce_sum(support_accs) / tf.to_float(args.meta_batch_size)
                self.metaval_total_accs2 = total_accs2 = [tf.reduce_sum(query_accs[j]) / tf.to_float(args.meta_batch_size) for j in range(num_updates)]

        tf.summary.scalar(prefix + 'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_acc1)

        for j in range(num_updates):
            tf.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix + 'Post-update accuracy, step ' + str(j+1), total_accs2[j])


    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden[0])):
            weights['w' + str(i + 1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, input_data, weights, reuse=False):
        hidden = normalize(tf.matmul(input_data, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i+1)]) + weights['b' + str(i+1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        return tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights['b' + str(len(self.dim_hidden) + 1)]



def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if args.datasource == 'momentum':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

    if args.log:
        train_writer = tf.summary.FileWriter(args.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, args.pretrain_iterations + args.metatrain_iterations):
        feed_dict = {}
        if itr < args.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[args.num_updates - 1]])
            if model.classification:
                input_tensors.extend([model.total_acc1, model.total_accs2[args.num_updates - 1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if args.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            if itr < args.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - args.pretrain_iterations)
            print_str += ": " + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr != 0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, args.logdir + '/' + exp_string + '/model' + str(itr))

        if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
            feed_dict = {}
            if model.classification:
                input_tensors = [model.metaval_total_acc1, model.metaval_total_accs2[args.num_updates - 1], model.summ_op]
            else:
                input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[args.num_updates - 1], model.summ_op]

            result = sess.run(input_tensors, feed_dict)
            print('Validation result: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, args.logdir + '/' + exp_string + '/model' + str(itr))

NUM_TEST_POINTS = 600

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes

    np.random.seed(1234)
    random.seed(1234)

    metaval_accs = []

    for _ in range(NUM_TEST_POINTS):
        feed_dict = {model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([model.metaval_total_acc1] + model.metaval_total_accs2, feed_dict)
        else:
            result = sess.run([model.total_loss1] + model.total_losses2, feed_dict)
        metaval_accs.append(result)

    metaval_accs = np.array(metaval_accs)
    means = np.mean(metaval_accs, 0)
    stds = np.std(metaval_accs, 0)
    ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = args.logdir + '/' + exp_string + '/' + 'test_ubs' + str(args.update_batch_size) + '_stepsize' + str(args.update_lr) + '.csv'
    out_pkl = args.logdir + '/' + exp_string + '/' + 'test_ubs' + str(args.update_batch_size) + '_stepsize' + str(args.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accs}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update' + str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)





def mata_main():
    if args.datasource == 'momentum':
        if args.train:
            test_num_updates = 5
        else:
            test_num_updates = 10

    if args.train is False:
        orig_meta_batch_size = args.meta_batch_size
        args.meta_batch_size = 1

    if args.datasource == 'momentum':
        data_generator = DataGenerator(args.update_batch_size * 2, args.meta_batch_size)
    else:
        data_generator = DataGenerator(args.update_batch_size * 2, args.meta_batch_size)


    dim_input = data_generator.dim_input
    dim_output = data_generator.dim_output

    if args.datasource == 'momentum':
        tf_data_load = True
        num_classes = data_generator.num_classes

        if args.train:
            random.seed(1234)
            data_tensor, label_tensor = data_generator.make_data_tensor()
            support_x = tf.slice(data_tensor, [0, 0, 0], [-1, num_classes * args.update_batch_size, -1])
            query_x = tf.slice(data_tensor, [0, num_classes * args.update_batch_size, 0], [-1, -1, -1])
            support_y = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * args.update_batch_size, -1])
            query_y = tf.slice(label_tensor, [0, num_classes * args.update_batch_size, 0], [-1, -1, -1])
            input_tensors = {'support_x': support_x, 'query_x': query_x, 'support_y': support_y, 'query_y': query_y}

        random.seed(12345)
        data_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        support_x = tf.slice(data_tensor, [0, 0, 0], [-1, num_classes * args.update_batch_size, -1])
        query_x = tf.slice(data_tensor, [0, num_classes * args.update_batch_size, 0], [-1, -1, -1])
        support_y = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * args.update_batch_size, -1])
        query_y = tf.slice(label_tensor, [0, num_classes * args.update_batch_size, 0], [-1, -1, -1])
        metaval_input_tensors = {'support_x': support_x, 'query_x': query_x, 'support_y': support_y, 'query_y': query_y}
    else:
        tf_data_load = False
        input_tensors = None


    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if args.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    sess = tf.InteractiveSession()

    if train is False:
        args.meta_batch_size = orig_meta_batch_size

    if args.train_update_batch_size == -1:
        args.train_update_batch_size = args.update_batch_size
    if args.train_update_lr == -1:
        args.train_update_lr = args.update_lr

    exp_string = 'cls_' + str(args.num_classes) + '.mbs_' + str(args.meta_batch_size) + \
                 '.ubs_' + str(args.train_update_batch_size) + '.numstep' + str(args.num_updates) + \
                 '.updatelr' + str(args.train_update_lr)


    if args.num_filters != 64:
        exp_string += 'hidden' + str(args.num_filters)
    if args.max_pool:
        exp_string += 'maxpool'
    if args.stop_grad:
        exp_string += 'stopgrad'
    if args.baseline:
        exp_string += args.baseline
    if args.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif args.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif args.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if args.resume or not args.train:
        model_file = tf.train.latest_checkpoint(args.logdir + '/' + exp_string)
        if args.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(args.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print('Restoring model weights from ' + model_file)
            saver.restore(sess, model_file)

    if args.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)
















is_train=True
def main(is_train=True):
    # idx_list = ['MSCIWORLD', 'MSCIEM', 'KOSPI200', 'KOSDAQ150', 'SPX500', 'NASDAQ100', 'RUSSELL2000',
    #             'CSI300', 'HSCE', 'TOPIX100', 'MSCIUSREITTR', 'USDKRW', 'GSCIGOLD', 'KISCOMPBONDCALL']

    idx_list = ['KOSPI200', 'KISCOMPBONDCALL']

    timesteps = 12  # 12 months data
    input_dim = 1   # monthly return

    # data processing
    df = get_data(idx_list)

    df_logp = df.pivot(index='eval_d', columns='idxcd', values='log_p')
    df_pivoted = df.pivot(index='eval_d', columns='idxcd', values='y')
    data_df = pd.DataFrame(columns=['idx_cd', 'date_0'] + ['t{}'.format(i - timesteps + 1) for i in range(timesteps + 1)])
    data_df_logp = pd.DataFrame(columns=['idx_cd', 'date_0'] + ['t{}'.format(i - timesteps + 1) for i in range(timesteps + 1)])
    n_date = len(df_pivoted.index)
    for idx in idx_list:
        df_idx = df_pivoted[idx]
        df_idx_logp = df_logp[idx]
        for k in range(timesteps, n_date):
            added_row = [i for i in df_idx[(k - timesteps): (k+1)]]
            added_row_logp = [i for i in df_idx_logp[(k - timesteps): k]] + [df_idx[k]]
            if np.sum(np.isnan(added_row)) == 0:
                data_df = data_df.append(pd.DataFrame([[idx, df_idx.index[k-1]] + added_row],
                                     columns=list(data_df.columns)), ignore_index=True)
            if np.sum(np.isnan(added_row_logp)) == 0:
                data_df_logp = data_df_logp.append(pd.DataFrame([[idx, df_idx_logp.index[k-1]] + added_row_logp],
                                     columns=list(data_df.columns)), ignore_index=True)

    assert data_df.shape == data_df_logp.shape

    tb_acc = TensorBoard(log_dir="./graph", histogram_freq=0, write_graph=True, write_images=True)
    early_stopping = EarlyStopping(patience=10)
    # training and backtesting
    columns_for_store = ['date_', 'port_y', 'port_bm_y', 'bm_all_y', 'bond_y', 'n_risk', 'bm_n_risk', 'bond_wgt', 'acc', 'bm_acc', 'precision', 'bm_precision', 'recall', 'bm_recall']
    portfolio_df = pd.DataFrame(columns=columns_for_store)
    constituent_df = pd.DataFrame(columns=['date_', 'port'] + idx_list)
    for i, t in enumerate(range(120, n_date - 1)):
        train_start_d, valid_start_d, test_d = df_pivoted.index[t - 120], df_pivoted.index[t - 24], df_pivoted.index[t]

        print('{}. {} started.'.format(i, test_d))
        # 첫번째 또는 12월말 기준 모델 재학습
        if (i == 0) or (datetime.datetime.strptime(test_d, '%Y-%m-%d').month == 12):
            # 12월말 기준 3년이상 레코드가 있는 자산들 선택 (1년이상 트레이닝 기간 / 2년 평가기간)
            universe = list()
            bond_y = 0.
            for idx in idx_list:
                if len(data_df[(data_df.idx_cd == idx) & (data_df.date_0 >= train_start_d) & (data_df.date_0 < test_d)]) >= 36:
                    if idx != 'KISCOMPBONDCALL':
                        universe.append(idx)

            print('{}. {} Universe selected.'.format(i, test_d))
            train_df = data_df[data_df.idx_cd.isin(universe) & (data_df.date_0 >= train_start_d) & (data_df.date_0 < valid_start_d)]
            valid_df = data_df[data_df.idx_cd.isin(universe) & (data_df.date_0 >= valid_start_d) & (data_df.date_0 < test_d)]

            train_df_logp = data_df_logp[data_df_logp.idx_cd.isin(universe) & (data_df_logp.date_0 >= train_start_d) & (data_df_logp.date_0 < valid_start_d)]
            valid_df_logp = data_df_logp[data_df_logp.idx_cd.isin(universe) & (data_df_logp.date_0 >= valid_start_d) & (data_df_logp.date_0 < test_d)]


            train_data_orig = np.array(train_df_logp.ix[:, 2:-1])
            train_data = (train_data_orig - np.expand_dims(np.mean(train_data_orig, axis=1), axis=1)) \
                         / np.expand_dims(np.std(train_data_orig, axis=1, ddof=1), axis=1)
            train_label = np.array(train_df_logp.ix[:, -1:] >= -0.00) * 1

            valid_data_orig = np.array(valid_df_logp.ix[:, 2:-1])
            valid_data = (valid_data_orig - np.expand_dims(np.mean(valid_data_orig, axis=1), axis=1)) \
                         / np.expand_dims(np.std(valid_data_orig, axis=1, ddof=1), axis=1)
            valid_label = np.array(valid_df_logp.ix[:, -1:] >= -0.00) * 1

            print('{}. {} Input Data Processed.'.format(i, test_d))
            with K.tf.device('/gpu:0'):
                input_p = Input(shape=(timesteps,), name='input_p')
                x = Dense(64, activation='relu', kernel_initializer='glorot_normal')(input_p)
                x = BatchNormalization()(x)
                x = Dense(32, activation='relu', kernel_initializer='glorot_normal')(x)
                x = BatchNormalization()(x)
                x = Dense(16, activation='linear', kernel_initializer='glorot_normal')(x)
                main_output = Dense(1, activation='sigmoid', name='output')(x)

                model = Model(inputs=[input_p], outputs=main_output)

                model.compile(optimizer='adam',
                              loss='binary_crossentropy',
                              metrics=['acc'])

                if is_train is True:
                    model.fit([train_data], train_label,
                              epochs=200,
                              batch_size=128,
                              shuffle=True,
                              class_weight={0: np.sum(train_label == 1), 1: np.sum(train_label == 0)},
                              validation_data=[[valid_data], valid_label],
                              callbacks=[tb_acc])  # starts training

                    print('{}. {} Final Model trained.'.format(i, test_d))

        test_df = data_df[data_df.idx_cd.isin(universe) & (data_df.date_0 == test_d)]
        test_data_y = np.array(test_df.ix[:, 2:-1])

        test_df_logp = data_df_logp[data_df_logp.idx_cd.isin(universe) & (data_df_logp.date_0 == test_d)]
        test_data_orig = np.array(test_df_logp.ix[:, 2:-1])
        test_data = (test_data_orig - np.expand_dims(np.mean(test_data_orig, axis=1), axis=1)) \
                    / np.expand_dims(np.std(test_data_orig, axis=1, ddof=1), axis=1)


        # bond_y = data_df[(data_df.idx_cd == 'KISCOMPBONDCALL') & (data_df.date_0 == test_d)].t1.values[0]
        bond_y = 0
        print('{}. {} AI Portfolio Result'.format(i, test_d))
        univ_long = list(test_df_logp[model.predict(test_data) >= 0.5].idx_cd)
        if len(univ_long) == 0:
            port_y = bond_y
        else:
            port_y = np.average(test_df_logp[model.predict(test_data) >= 0.5].t1) * len(univ_long) / len(universe) \
                     + bond_y * (1 - len(univ_long) / len(universe))
        acc, precision, recall = get_accuracy((model.predict(test_data) >= 0.5).reshape([-1]),
                                              np.array(test_df_logp.t1 >= 0))

        print('{}. {} BM Portfolio Result'.format(i, test_d))
        univ_bm_long = list(test_df_logp[np.prod(1 + test_data_y, axis=1) >= 1].idx_cd)
        if len(univ_bm_long) == 0:
            port_bm_y = bond_y
        else:
            port_bm_y = np.average(test_df[np.prod(1 + test_data_y, axis=1) >= 1].t1) * len(univ_bm_long) / len(universe) \
                     + bond_y * (1 - len(univ_bm_long) / len(universe))
        bm_acc, bm_precision, bm_recall = get_accuracy((np.prod(1 + test_data_y, axis=1) >= 1).reshape([-1]),
                                              np.array(test_df.t1 >= 0))

        bm_all_y = np.average(test_df.t1)

        print('{}. {} appended to the result'.format(i, test_d))
        portfolio_df = portfolio_df.append(pd.DataFrame([[test_d, port_y, port_bm_y,
                                                          bm_all_y, bond_y,
                                                          len(univ_long), len(univ_bm_long),
                                                          (1 - len(univ_long) / len(universe)), acc, bm_acc,
                                                          precision, bm_precision, recall, bm_recall]],
                                                        columns=columns_for_store), ignore_index=True)

        const_arr = np.zeros(len(idx_list))
        const_arr[list(map(idx_list.index, univ_long))] = 1
        constituent_df = constituent_df.append(
            pd.DataFrame([[test_d, 'strat'] + list(const_arr)], columns=['date_', 'port'] + idx_list),
            ignore_index=True)

        const_arr = np.zeros(len(idx_list))
        const_arr[list(map(idx_list.index, univ_bm_long))] = 1
        constituent_df = constituent_df.append(
            pd.DataFrame([[test_d, 'bm'] + list(const_arr)], columns=['date_', 'port'] + idx_list),
            ignore_index=True)

    print('plot..')
    get_result(portfolio_df)
    mill_df = portfolio_df[portfolio_df.date_ >= '2000-01-01'].reset_index()
    get_result(mill_df)

    model.save('cardiff_model.h5')

    print('save..')
    portfolio_df.to_csv("cardiff_lstm_outsample_tsmom_kospi200.csv")
    # portfolio_df.to_csv("cardiff_lstm_outsample_tsmom_mtl_balanced.csv")
    # constituent_df.to_csv("cardiff_lstm_outsample_tsmom_mtl_balanced_constituent.csv")
    print('successfully stored.')





