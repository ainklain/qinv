
import tensorflow as tf
import numpy as np
import pandas as pd
import time

np.random.seed(1234)


def data_generator(load=True):
    if load is True:
        log_y = pd.read_csv('sample_log_return.csv', index_col=0).values
        n_asset = 28
        assert log_y.shape[1] % n_asset == 0
        n_scenario = log_y.shape[0]
        T = log_y.shape[1] // n_asset
        n_batch = n_scenario

        log_y = np.transpose(log_y.reshape(n_scenario, n_asset, T), axes=(0, 2, 1))

    else:
        n_scenario = 1000
        n_batch = 1000

        mu = np.array([0.05, 0.01, 0.08, 0.03, 0.02, 0.09, 0.02])
        sig = np.array([0.15, 0.03, 0.26, 0.11, 0.07, 0.31, 0.03])
        T = 50
        n_asset = len(mu)

        log_y = np.zeros([n_scenario, T, n_asset])
        for i, (m, s) in enumerate(zip(mu, sig)):
            for t in range(T):
                z = np.random.randn(n_scenario)
                log_y[:, t, i] = (m - 0.5 * s ** 2) * 1 + s * z * np.sqrt(1)

    return log_y, n_scenario, n_asset, n_batch, T


def cagr(rate_, t):
    return (1. + rate_) ** t - 1.


def log_cagr(rate_, t):
    return t * np.log(1. + rate_)


# Data
log_y, n_scenario, n_asset, n_batch, T = data_generator(load=True)
target_r = log_cagr(0.05, 10)

# n_batch = 1000

# cumulative log return
y_index = np.zeros([n_batch, T, n_asset])
for t in range(T):
    y_index[:, t, :] = np.exp(np.sum(log_y[:n_batch, :(t+1), :], axis=1))

# Model
X = tf.placeholder(tf.float32, [None, T, n_asset], 'x')    # log return
# w = tf.Variable(tf.random_normal(shape=[1, n_asset, 1, 1], mean=1./n_asset, stddev=0.5/n_asset), name='w')
w = tf.Variable(tf.reshape(tf.constant([0.0502917,0.00094968,0.00171818,0.00055809,0.00796205,0.00429873,0.00614835,0.00210668,0.00286694,0.00286965,0.04644709,0.00044132,0.10670789,0.01428658,0.05973064,0.1582465,0.00236258,0.00507528,0.00498536,0.199976,0.05046877,0.18661235,0.00305916,0.0046163,0.00045414,0.00438564,0.07144486,0.0009383]), [1, 28, 1, 1]), name='w')

# w = tf.Variable(tf.reshape(tf.constant([0.0496179752,0.000001922206,0.00086026717,0,0.0241905942,0.00438336058,0.0113709593,0.000105731063,0.0020191342,0.000043880235,0.0487454672,0.000214151805,0.177880423,0.00518260542,0.0389312503,0.189835885,0.00375898557,0.000214993531,0.00215865315,0.199832517,0.0331084066,0.160659875,0.0153625943,0.00405513809,0.000667626166,0.00170694946,0.0250887144,0.000001941124]), [1, 28, 1, 1]), name='w')

with tf.variable_scope('layer'):
    X_reshaped = tf.reshape(X, [1, -1, n_asset, 1])
    l1 = tf.nn.conv2d(X_reshaped, w, strides=[1, 1, 1, 1], padding='VALID')
    l1 = tf.reshape(l1, [1, -1, T])
    l1_y = tf.slice(l1, [0, 0, 1], [1, -1, T-1]) / tf.slice(l1, [0, 0, 0], [1, -1, T-1]) - 1.

    cvar_pt = tf.contrib.distributions.percentile(l1_y, q=5.)
    cvar = tf.reduce_mean(tf.boolean_mask(l1_y, l1_y <= cvar_pt))

    l2 = tf.subtract(tf.squeeze(tf.slice(l1, [0, 0, T-1], [1, -1, 1])), tf.constant(target_r, dtype=tf.float32))

    l3 = tf.nn.sigmoid(l2)
    out = tf.reduce_mean(l3)


with tf.variable_scope('loss'):
    loss = - tf.log(out) \
           + tf.squared_difference(0.9999, tf.reduce_sum(w)) * 10000 \
           + tf.reduce_sum(tf.maximum(-w, 0.)) * 100 \
           + tf.reduce_sum(tf.maximum(w - 0.2, 0.)) * 100 \
           + tf.reduce_sum(tf.maximum(-cvar - 0.05, 0)) * 100 \
           + tf.maximum(-tf.reduce_sum(tf.squeeze(w) * tf.constant([1] + [-1 for i in range(9)] + [0 for i in range(18)], dtype=tf.float32)), 0)  * 100 \
           + tf.maximum(0.05-tf.reduce_sum(tf.squeeze(w) * tf.constant([1] + [0 for i in range(25)] + [0.05 for i in range(2)], dtype=tf.float32)), 0)* 100 \
           + tf.maximum(0.05-tf.reduce_sum(tf.squeeze(w) * tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.05], dtype=tf.float32)), 0)* 100

with tf.variable_scope('optim'):
    optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


# Session initialization
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

losses = list()
w_list = pd.DataFrame(columns=np.arange(n_asset))
# shuffled_idx = np.random.choice(len(y), size=len(y), replace=False)
# x_data = y[shuffled_idx].reshape([1, n_scenario, n_asset, 1])


# pre-train

s_t = time.time()
min_loss = 99999
count_loss = 0
before_t = time.time()
x_data = y_index[:100]
for i in range(50000):
    _, loss_val, w_val, cvar_val = sess.run([optim, loss, w, cvar], feed_dict={X: x_data})
    if i % 100 == 0:
        now_t = time.time()
        print("i: {} // time: {} == loss: {:.5f} // sum_w: {} // cvar: {}".format(
            i, now_t - before_t, loss_val, np.sum(w_val), cvar_val))
        before_t = now_t
        if min_loss <= loss_val:
            count_loss = count_loss + 1
        else:
            min_loss = loss_val
            min_w = w_val.squeeze()
            count_loss = 0

    if count_loss >= 20:
        print('early stopped')
        break

e_t = time.time()
print("#### Pretrain total time: {}\nmin_w : \n{}".format(e_t - s_t, min_w))

# train

s_t = time.time()
min_loss = 99999
count_loss = 0
before_t = time.time()
x_data = y_index[100:10000]
for i in range(50000):
    _, loss_val, w_val, cvar_val = sess.run([optim, loss, w, cvar], feed_dict={X: x_data})
    if i % 100 == 0:
        now_t = time.time()
        print("i: {} // time: {} == loss: {:.5f} // sum_w: {} // cvar: {} \n w: {}".format(
            i, now_t - before_t, loss_val, np.sum(w_val), cvar_val, w_val.squeeze()))
        before_t = now_t
        if min_loss <= loss_val:
            count_loss = count_loss + 1
        else:
            min_loss = loss_val
            min_w = w_val.squeeze()
            count_loss = 0

    if count_loss >= 10:
        print('early stopped')
        break

e_t = time.time()
print("#### total time: {}\nmin_w : \n{}".format(e_t - s_t, min_w))

# for bat_num in range(10):
#     x_data = y[np.random.choice(len(y), size=n_batch, replace=False)].reshape([1, n_batch, n_asset, 1])
#     min_loss = 99999
#     count_loss = 0
#     for i in range(50000):
#         _, loss_val, w_val = sess.run([optim, loss, w], feed_dict={X: x_data})
#         if i % 50 == 0:
#             print("bat_num:{} || i: {} == loss: {:.5f} // w: {} // sum_w: {}".format(bat_num, i, loss_val, w_val.squeeze(), np.sum(w_val)))
#             if min_loss <= loss_val:
#                 count_loss = count_loss + 1
#             else:
#                 min_loss = loss_val
#                 count_loss = 0
#
#         if count_loss >= 20:
#             break
#
#     losses.append(loss_val)
#     w_list = w_list.append(pd.Series(w_val.squeeze()), ignore_index=True)

sess.close()


w_list.to_csv('optim_result.csv')