
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
        # log_y shape: [n_scenario, T, n_asset]
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
    return (1. + rate_) ** t


def log_cagr(rate_, t):
    return t * np.log(1. + rate_)


# Data
log_y, n_scenario, n_asset, n_batch, T = data_generator(load=True)
# target_r = log_cagr(0.07, 10)
target_r = cagr(0.06, 10)

# n_batch = 1000

# cumulative log return
y_index = np.ones([n_batch, T+1, n_asset])
for t in range(1, T+1):
    y_index[:, t, :] = np.exp(np.sum(log_y[:n_batch, :t, :], axis=1))

# Model
X = tf.placeholder(tf.float32, [None, T+1, n_asset], 'x')    # index of returns
# w = tf.Variable(tf.random_normal(shape=[1, n_asset, 1, 1], mean=1./n_asset, stddev=0.5/n_asset), name='w')
# w = tf.Variable(tf.reshape(tf.constant([0.0502917,0.00094968,0.00171818,0.00055809,0.00796205,0.00429873,0.00614835,0.00210668,0.00286694,0.00286965,0.04644709,0.00044132,0.10670789,0.01428658,0.05973064,0.1582465,0.00236258,0.00507528,0.00498536,0.199976,0.05046877,0.18661235,0.00305916,0.0046163,0.00045414,0.00438564,0.07144486,0.0009383]), [1, 28, 1, 1]), name='w')
w = tf.Variable(tf.reshape(tf.constant([0.051619951,0.00056610547,0.0046637878,0.00044136148,0.025633262,0.0054883324,0.0051823771,0.000033788681,0.0030320571,0.00005498137,0.050232582,0.0033770176,0.19533594,0.0042381487,0.08769431,0.19816957,0.0022777577,0.00013492002,0.0017327978,0.19954728,0.021804467,0.075694643,0.045952223,0.0017005004,0.0043201558,0.0052067977,0.00090890261,0.0049188263]), [1, 28, 1, 1]), name='w')

# w = tf.Variable(tf.reshape(tf.constant([0.0496179752,0.000001922206,0.00086026717,0,0.0241905942,0.00438336058,0.0113709593,0.000105731063,0.0020191342,0.000043880235,0.0487454672,0.000214151805,0.177880423,0.00518260542,0.0389312503,0.189835885,0.00375898557,0.000214993531,0.00215865315,0.199832517,0.0331084066,0.160659875,0.0153625943,0.00405513809,0.000667626166,0.00170694946,0.0250887144,0.000001941124]), [1, 28, 1, 1]), name='w')
# w = tf.Variable(tf.reshape(tf.constant([0.05115083, 0.00158095, 0.00188144, 0.00250211, 0.0276066,  0.00298387, 0.00428348, 0.00024822, 0.00132392, 0.00059819, 0.05238885, 0.001516, 0.19971567, 0.00244328, 0.11267529, 0.1979491,  0.00130896, 0.00368782, 0.00136,    0.19960612, 0.00445453, 0.07429556, 0.03972487, 0.00453786, 0.00154942, 0.00594129, 0.00223093, 0.0003635]), [1, 28, 1, 1]), name='w')
# cvar1, out1, loss1 = sess.run([cvar, out, loss], feed_dict={X: y_index})
# print("cvar:{}  out:{}  loss:{}".format(cvar1, out1, loss1))

# sess.run(tf.reduce_sum(tf.cast(l2 >= 0, tf.float32)), feed_dict={X: y_index})

with tf.variable_scope('layer'):
    X_reshaped = tf.reshape(X, [1, -1, n_asset, 1])
    l1 = tf.nn.conv2d(X_reshaped, w, strides=[1, 1, 1, 1], padding='VALID')
    l1 = tf.reshape(l1, [1, -1, T+1])
    l1_y = tf.slice(l1, [0, 0, 1], [1, -1, T]) / tf.slice(l1, [0, 0, 0], [1, -1, T]) - 1.

    cvar_pt = tf.contrib.distributions.percentile(l1_y, q=5.)
    cvar = tf.reduce_mean(tf.boolean_mask(l1_y, l1_y <= cvar_pt))

    l2 = tf.subtract(tf.squeeze(tf.slice(l1, [0, 0, T], [1, -1, 1])), tf.constant(target_r, dtype=tf.float32))

    l3 = tf.nn.sigmoid(l2)
    out = tf.reduce_mean(l3)

    acc = tf.reduce_sum(tf.cast(l2 >= 0, tf.float32))


    # tf.l3

with tf.variable_scope('loss'):
    loss = - tf.log(out) \
           + tf.squared_difference(0.9999, tf.reduce_sum(w)) * 10000 \
           + tf.reduce_sum(tf.maximum(-w, 0.)) * 100 \
           + tf.reduce_sum(tf.maximum(w - 0.2, 0.)) * 100 \
           + tf.reduce_sum(tf.maximum(-cvar - 0.06, 0)) * 100 \
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


filename = 'optim_result.txt'

# pre-train
s_t = time.time()
min_loss = 99999
count_loss = 0
before_t = time.time()
x_data = y_index[:100]
for i in range(100000):
    _, loss_val, out_val, w_val, cvar_val, acc_val = sess.run([optim, loss, out, w, cvar, acc], feed_dict={X: x_data})
    if i % 100 == 0:
        now_t = time.time()
        print("i: {} (max count:{}) // time: {} == loss: {:.5f} // out: {} // sum_w: {} // cvar: {} // acc: {}".format(
            i, count_loss, now_t - before_t, loss_val, out_val, np.sum(w_val), cvar_val, acc_val / len(x_data)))
        before_t = now_t
        if min_loss <= loss_val:
            count_loss = count_loss + 1
        else:
            min_loss = loss_val
            min_out = out_val
            min_cvar = cvar_val
            min_acc = acc_val / len(x_data)
            min_w = w_val.squeeze()
            count_loss = 0

    # if count_loss >= 30:
    #     print('early stopped')
    #     break

e_t = time.time()
print_txt = "#### Pretrain total time: {}  // loss: {} // acc:{} // out: {}  // cvar: {}\nmin_w : \n{}\n\n".format(
    e_t - s_t, min_loss, min_acc, min_out, min_cvar, min_w)
print(print_txt)

with open(filename, 'w') as myfile:
    myfile.write(print_txt)

# train

s_t = time.time()
min_loss = 99999
count_loss = 0
before_t = time.time()
x_data = y_index[:10000]
for i in range(50000):
    _, loss_val, out_val, w_val, cvar_val, acc_val = sess.run([optim, loss, out, w, cvar, acc], feed_dict={X: x_data})
    if i % 100 == 0:
        now_t = time.time()
        print("i: {} (max count:{}) // time: {} == loss: {:.5f} // out: {} // sum_w: {} // cvar: {} // acc: {}".format(
            i, count_loss, now_t - before_t, loss_val, out_val, np.sum(w_val), cvar_val, acc_val / len(x_data)))
        before_t = now_t
        if min_loss <= loss_val:
            count_loss = count_loss + 1
        else:
            min_loss = loss_val
            min_out = out_val
            min_cvar = cvar_val
            min_acc = acc_val / len(x_data)
            min_w = w_val.squeeze()
            count_loss = 0

    if i % 10000 == 0:
        print_txt2 = "i: {} (max count:{}) // time: {} == loss: {:.5f} // acc: {} // out: {} // sum_w: {} // cvar: {}\nmin_w: {}\n\n".format(
            i, count_loss, now_t - before_t, min_loss, min_acc, min_out, np.sum(w_val), min_cvar, min_w)
        with open(filename, 'a') as myfile:
            myfile.write(print_txt2)

    if count_loss >= 20:
        print('early stopped')
        break

e_t = time.time()
print_txt3 = "#### total time: {}  // loss: {} // acc: {} //  out: {}  // cvar: {}\nmin_w : \n{}".format(
    e_t - s_t, min_loss, min_acc, min_out, min_cvar, min_w)
print(print_txt3)

with open(filename, 'a') as myfile:
    myfile.write(print_txt3)
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