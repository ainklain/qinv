
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.platform import flags
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.ops.init_ops import glorot_uniform_initializer
import pickle


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'momentum', 'Which dataset to work on')
flags.DEFINE_string('architecture', 'mom_model', 'Which network architecture to use')
flags.DEFINE_integer('n_input', 33, 'Number of nodes in the input layer')
flags.DEFINE_integer('n_hidden_1', 40, 'Number of nodes in the first auto encoder hidden layer')
flags.DEFINE_integer('n_hidden_2', 30, 'Number of nodes in the second auto encoder hidden layer')
flags.DEFINE_integer('n_hidden_3', 50, 'Number of nodes in the fully connected layer')
flags.DEFINE_integer('n_output', 2, 'Number of nodes in the output layer')
flags.DEFINE_integer('n_epoch_ae', 10, 'Max Number of epochs to train autoencoder')
flags.DEFINE_integer('n_epoch_all', 10, 'Max Number of epochs to train all graph')
flags.DEFINE_integer('batch_size', 256, 'Number of samples per batch')
flags.DEFINE_float('learning_rate', 0.0002, 'Leearning rate')
flags.DEFINE_integer('patience', 10, 'Patience count for early stopping')


f = open('fundamentals.pkl', 'rb')
data_pickle = pickle.load(f)
f_data = data_pickle['data']
f_label = data_pickle['label']


train_data = tf.data.Dataset.from_tensor_slices(f_data[:int(len(f_data) * 0.8)])
train_label = tf.data.Dataset.from_tensor_slices(f_label[:int(len(f_label) * 0.8)])
train_dataset = tf.data.Dataset.zip((train_data, train_label)).shuffle(10000).repeat().batch(FLAGS.batch_size)

valid_data = tf.data.Dataset.from_tensor_slices(f_data[int(len(f_data) * 0.8):])
valid_label = tf.data.Dataset.from_tensor_slices(f_label[int(len(f_label) * 0.8):])
valid_dataset = tf.data.Dataset.zip((valid_data, valid_label)).shuffle(10000).repeat().batch(FLAGS.batch_size)

# iterator = train_dataset.make_initializable_iterator()
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
next_element = iterator.get_next()

train_init_op = iterator.make_initializer(train_dataset)
valid_init_op = iterator.make_initializer(valid_dataset)


def fullyconnected(in_data, is_training=False):
    with tf.name_scope('fc'):
        net = tf.layers.dense(in_data, 1024, tf.nn.relu)
        net = tf.layers.dropout(net, rate=0.5, training=is_training)
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dense(net, 1024, tf.nn.relu)
        net = tf.layers.dropout(net, rate=0.5, training=is_training)
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dense(net, 16, tf.nn.relu)
    return net


predicted = fullyconnected(next_element[0])

with tf.name_scope('loss'):
    loss = tf.losses.mean_squared_error(labels=next_element[1], predictions=predicted)
tf.summary.scalar('loss', loss)

optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

# prediction = tf.argmax(logits, 1)
# equality = tf.equal(prediction, tf.argmax(next_element[1], 1))
# with tf.name_scope('accuracy'):
#     accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
# tf.summary.scalar('accuracy', accuracy)

init_op = tf.global_variables_initializer()
patience_cnt = 0
batch_per_ep = int(len(f_label) * 0.8) // FLAGS.batch_size
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train', sess.graph)
    # valid_writer = tf.summary.FileWriter('./valid')

    sess.run(init_op)
    sess.run(train_init_op)
    # for i in range(epochs):
    prev_l = 9999
    loss_ep = list()
    for ep in range(FLAGS.n_epoch_ae):
        loss_avg = 0
        for i in range(batch_per_ep):
            summary, l, _ = sess.run([merged, loss, optimizer])
            train_writer.add_summary(summary, i)
            if i % 200 == 0:
                print('[Train AE] Epoch: {}, batch: {}, loss: {:.3f}'.format(ep, i, l))
                if prev_l - l > 0.005:
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                prev_l = l
                if patience_cnt > FLAGS.patience:
                    print('early stopping..')
                    break


    train_iters = 100
    sess.run(train_init_op)
    avg_acc = 0
    for i in range(train_iters):
        acc = sess.run([accuracy])
        avg_acc += acc[0]
    print("Average training set accuracy over {} iteration is {:.2f}%".format(train_iters, (avg_acc / train_iters) * 100))

    valid_iters = 100
    sess.run(valid_init_op)
    avg_acc = 0
    for i in range(valid_iters):
        acc = sess.run([accuracy])
        avg_acc += acc[0]
    print("Average validation set accuracy over {} iteration is {:.2f}%".format(valid_iters, (avg_acc / valid_iters) * 100))









# class MyModel(ModelDefault):
#     def __init__(self, univ=None, sch=None, **kwargs):
#         super().__init__(univ, sch, **kwargs)
#
#     def value(self):
#         mktcap = self.equity_obj['mktcap']
#         be = self.equity_obj['be']
#         bm = be / mktcap
#
#         factor_info = {'name': 'value',
#                        'item': {'bm': bm},
#                        'long_short': 'LS',
#                        'min_n_of_stocks': 10}
#         result = self.factor_write(**factor_info)
#
#     def quality(self):
#         gpa = self.equity_obj['gpa']
#
#         factor_info = {'name': 'quality',
#                        'item': {'gpa': gpa},
#                        'winsorize': 0.4}
#         result = self.factor_write(**factor_info)
#
#     def size(self):
#         mktcap = self.equity_obj['mktcap']
#         close_ = self.equity_obj['close_']
#
#         factor_info = {'name': 'size',
#                        'item': {'mktcap': mktcap, 'close_': close_},
#                        'long_short': 'L',
#                        'q': 0.5,
#                        'reverse': True,
#                        'store_result': True,
#                        'overwrite_pipe': True}
#         self.set_pipe_by_info(**factor_info)
#         # mktcap = self.pipe.get_item('pipe_size', 'mktcap')
#         # close_= self.pipe.get_item('pipe_size', 'clse_')
#
#
