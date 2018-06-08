import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

learning_rate = 0.005
num_steps = 30000
batch_size = 256

display_step = 1000

num_hidden_1 = 40
num_hidden_2 = 30
num_input = 33

np.random.seed(1234)

# x_train = np.array([[1, 2, 3, 4, 5],
#                    [2, 3, 4, 1, 3],
#                    [3, 2, 4, 5, 6],
#                    [2, 3, 4, 5, 1],
#                    [9, 6, 3, 5, 5],
#                    [4, 3, 2, 6, 5],
#                    [3, 2, 5, 4, 3]])
x_train= np.random.random([200, num_input])
x_train_norm = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)


X = tf.placeholder("float", [None, num_input])


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


weights = {
    'encoder_h1': tf.Variable(xavier_init(num_input, num_hidden_1)),
    'encoder_h2': tf.Variable(xavier_init(num_hidden_1, num_hidden_2)),
    'decoder_h1': tf.Variable(xavier_init(num_hidden_2, num_hidden_1)),
    'decoder_h2': tf.Variable(xavier_init(num_hidden_1, num_input)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

def encoder(x):
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2'])
    return layer_2


encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(1, num_steps + 1):
#         batch_x = x_train_norm
#         # Run optimization op (backprop) and cost op (to get loss value)
#         _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
#         # Display logs per step
#         if i % display_step == 0 or i == 1:
#             print('Step %i: Minibatch Loss: %f' % (i, l))
#     batch_test = x_train_norm[:3]
#     g = sess.run(decoder_op, feed_dict={X: batch_test})
print("{}\n{}".format(batch_test[0], g[0]))

f= open('mom.pickle', 'rb')
data_pickle = pickle.load(f)
mom_data = data_pickle['data']
mom_label = data_pickle['label']
mom_label_onehot = np.array([[1, 0] if i == 1 else [0, 1] for i in mom_label])

n_sample = mom_label_onehot.shape[0]
train_data = mom_data[:int(0.8 * n_sample)]
train_label = mom_label_onehot[:int(0.8 * n_sample)]
n_train = train_label.shape[0]

reshuffle_t = n_train // batch_size

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(1, num_steps+1):
        idx_pointer = (i - 1) % reshuffle_t
        if idx_pointer == 0:
            idx = np.arange(n_train)
            np.random.shuffle(idx)

        batch_x = mom_data[idx_pointer * batch_size:(idx_pointer+1) * batch_size]

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    batch_test = mom_data[:3]
    g = sess.run(decoder_op, feed_dict={X: batch_test})