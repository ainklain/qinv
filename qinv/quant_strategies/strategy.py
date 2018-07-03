from qinv.schedule import Schedule
from qinv.universe import Universe
from qinv.model import ModelDefault
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
from tqdm import tqdm
import pickle

epoch_num = 5
batch_size = 256

n_hidden_1 = 40
n_hidden_2 = 4
n_hidden_3 = 50

n_input = 33
n_output = 2

learning_rate = 0.002

class TFModel(ModelDefault):
    def __init__(self, univ=None, sch=None, **kwargs):
        super().__init__(univ, sch, **kwargs)

    def set_data(self, name='cnn_test'):
        if name == 'cnn_test':
            mktcap = self.equity_obj['mktcap']
            be = self.equity_obj['be']
            bm = be / mktcap

            gpa = self.equity_obj['gpa']
            close_ = self.equity_obj['close_']
            close_adj = self.equity_obj['close_adj']

            factor_info = {'name': name,
                           'item': {'mktcap': mktcap,
                                    'bm': bm,
                                    'gpa': gpa,
                                    'close_': close_,
                                    'close_adj': close_adj},
                           'store_result': True,
                           'overwrite_pipe': True,
                           }
        elif name == 'mom_data':
            close_ = self.equity_obj['close_']
            close_adj = self.equity_obj['close_adj']

            factor_info = {'name': name,
                           'item': {'close_': close_,
                                    'close_adj': close_adj},
                           'mode': 'load_or_run',
                           'chunksize': 10000, }

        self.set_pipe_by_info(**factor_info)
#
#
#
# # m = TFModel()
# # m.pipe.load('mom_test', usr='solution_research')
#
m = TFModel(univ=Universe(**{'equity':['us_all']}),
            sch=Schedule('1985-01-01', '2016-12-31', type_='end', freq_='m'))
m.set_data('mom_data')
# m.pipe.load('mom_test')
# close_ = m.pipe.get_item('mom_test', item_id='close_')
# close_adj = m.pipe.get_item('mom_test', item_id='close_adj')
#
# close_adj['y'] = close_adj.groupby('infocode')['value_'].transform(pd.Series.pct_change)
# close_adj['next_y'] = close_adj.groupby('infocode')['y'].shift(-1)
# returns = close_adj.drop(columns=['item_nm', 'value_'])
# returns = returns[~returns['y'].isnull()]
#
# sch = sorted(list(set(close_['eval_d'])))
#
#
# from dbmanager import SqlManager
# sqlm = SqlManager()
# sqlm.set_db_name('qinv')
#
# mom_data = None
# mom_label = None
# for i in tqdm(range(12, len(sch)), mininterval=5):
#     t = sch[i]
#     t_12 = sch[i-12]
#
#     if time.strptime(t, '%Y-%m-%d').tm_mon == 12:
#         jan_flag = 1
#     else:
#         jan_flag = 0
#
#     return_daily_all = sqlm.db_read("""
#     select infocode, str(rownum) as eval_d
#     , sum(log(1+y)) over (partition by infocode order by rownum rows between unbounded preceding and current row) as y
# 	 from (
# 		 select  infocode, marketdate as eval_d, y, ROW_NUMBER() over (partition by infocode order by marketdate desc) as rownum
# 		 from qinv..EquityReturnDaily
# 		 where marketdate > dateadd(day, -100, '{t}') and marketdate <= '{t}'
# 		 ) A
# 		 where rownum <= 20""".format(t=t))
#
#     cond_sch = np.array(close_['eval_d'] == t)
#     cond_ex_coin = np.array(close_['value_'] >= 5)
#     univ_t = close_[cond_sch & cond_ex_coin][['infocode', 'eval_d']]
#
#     return_daily = pd.merge(univ_t.loc[:, ['infocode']], return_daily_all, how='inner', on=['infocode'])
#     returns_t = returns[(returns['eval_d'] >= t_12) & (returns['eval_d'] <= t)]
#     returns_t.loc[:, ['y']] = returns_t['y'].apply(lambda _x: np.log(1 + _x))
#     returns_t.loc[:, ['y']] = returns_t.groupby(by=['infocode'])['y'].cumsum()
#
#     x = pd.merge(univ_t.loc[:, ['infocode']], returns_t.loc[returns_t['eval_d'] < t, ['infocode', 'eval_d', 'y']]
#                  , how='inner', on=['infocode'])
#     y = pd.merge(univ_t, returns_t[['infocode', 'eval_d', 'next_y']], how='inner', on=['infocode', 'eval_d'])
#
#     x_all = pd.concat([x, return_daily]).sort_values(by=['infocode', 'eval_d'])
#
#     x_pivoted = x_all.pivot(index='infocode', columns='eval_d', values='y')
#     x_pivoted_not_null = x_pivoted[x_pivoted.isna().sum(axis=1) == 0]
#     y_not_null = y[~y['next_y'].isna()]
#     x_normalized = (x_pivoted_not_null - x_pivoted_not_null.mean(axis=0)) / x_pivoted_not_null.std(axis=0)
#     data_w_label = pd.merge(x_normalized, y_not_null.loc[:, ['infocode', 'next_y']], on=['infocode'])
#     data_w_label['flag'] = jan_flag
#
#     if len(data_w_label) == 0:
#         # 데이터 없음
#         continue
#
#     label = np.array(data_w_label['next_y'])
#     label[label >= np.percentile(label, q=50)] = 1
#     label[label < np.percentile(label, q=50)] = 0
#
#     data = np.array(data_w_label.drop(columns=['infocode', 'next_y']))
#     if data.shape[1] < 33:
#         continue
#
#     if mom_label is None:
#         mom_label = label.copy()
#     else:
#         mom_label = np.concatenate([mom_label, label])
#
#     if mom_data is None:
#         mom_data = data.copy()
#     else:
#         mom_data = np.concatenate([mom_data, data])
#
#
# f= open('mom.pickle', 'wb')
# pickle.dump({'data': mom_data, 'label': mom_label}, f)

f= open('mom.pickle', 'rb')
data_pickle = pickle.load(f)
mom_data = data_pickle['data']
mom_label = data_pickle['label']


train_data = tf.data.Dataset.from_tensor_slices(mom_data[:int(len(mom_data) * 0.8)])
train_label = tf.data.Dataset.from_tensor_slices(mom_label[:int(len(mom_label) * 0.8)]).map(
    lambda z: tf.one_hot(tf.cast(z, tf.int32), 2))
train_dataset = tf.data.Dataset.zip((train_data, train_label)).shuffle(10000).repeat().batch(batch_size)

valid_data = tf.data.Dataset.from_tensor_slices(mom_data[int(len(mom_data) * 0.8):])
valid_label = tf.data.Dataset.from_tensor_slices(mom_label[int(len(mom_label) * 0.8):]).map(
    lambda z: tf.one_hot(tf.cast(z, tf.int32), 2))
valid_dataset = tf.data.Dataset.zip((valid_data, valid_label)).shuffle(10000).repeat().batch(batch_size)

# iterator = train_dataset.make_initializable_iterator()
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
next_element = iterator.get_next()

train_init_op = iterator.make_initializer(train_dataset)
valid_init_op = iterator.make_initializer(valid_dataset)

def nn_model(in_data):
    bn = tf.layers.batch_normalization(in_data)
    fc1 = tf.layers.dense(bn, 50)
    fc2 = tf.layers.dense(fc1, 50)
    # fc2 = tf.layers.dropout(fc2)
    fc3 = tf.layers.dense(fc2, 2)
    return fc3

def encoder(in_data):
    # bn = tf.layers.batch_normalization(in_data)
    enc1 = tf.layers.dense(in_data, n_hidden_1, tf.nn.tanh)
    encoded = tf.layers.dense(enc1, n_hidden_2, tf.nn.tanh)
    return encoded

def decoder(in_data):
    dec1 = tf.layers.dense(in_data, n_hidden_1, tf.nn.tanh)
    decoded = tf.layers.dense(dec1, n_input, tf.nn.tanh)
    return decoded

def fullyconnected(in_data):
    net = tf.layers.dense(in_data, n_hidden_3, tf.nn.relu)
    logits = tf.layers.dense(net, n_output)
    return logits

decoded = decoder(encoder(next_element[0]))
logits = fullyconnected(encoder(next_element[0]))

with tf.name_scope('loss_ae'):
    loss_ae = tf.losses.mean_squared_error(labels=next_element[0], predictions=decoded)
tf.summary.scalar('loss_ae', loss_ae)

with tf.name_scope('loss_all'):
    loss_all = tf.losses.mean_squared_error(labels=next_element[1], predictions=logits)
tf.summary.scalar('loss_all', loss_all)


optimizer_ae = tf.train.AdamOptimizer(learning_rate).minimize(loss_ae)
optimizer_all = tf.train.AdamOptimizer(learning_rate).minimize(loss_all)

# logits = nn_model(next_element[0])
# loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_element[1], logits=logits))
# optimizer = tf.train.AdamOptimizer().minimize(loss)


# prediction = tf.argmax(decoded, 1)
prediction = tf.argmax(logits, 1)
equality = tf.equal(prediction, tf.argmax(next_element[1], 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
tf.summary.scalar('accuracy', accuracy)

init_op = tf.global_variables_initializer()

batch_per_ep = int(len(mom_label) * 0.8) // batch_size
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train', sess.graph)
    # valid_writer = tf.summary.FileWriter('./valid')

    sess.run(init_op)
    sess.run(train_init_op)
    # for i in range(epochs):
    for ep in range(epoch_num):
        for i in range(batch_per_ep):
            summary, l, _ = sess.run([merged, loss_ae, optimizer_ae])
            train_writer.add_summary(summary, i)
            if i % 100 == 0:
                print('[Train AE] Epoch: {}, batch: {}, loss: {:.3f}'.format(ep, i, l))

    for ep in range(epoch_num):
        for i in range(batch_per_ep):
            summary, l, _, acc = sess.run([merged, loss_all, optimizer_all, accuracy])
            if i % 100 == 0:
                print('[Train All] Epoch: {}, batch: {}, loss: {:.3f}, training accuracy: {:.2f}%'.format(ep, i, l, acc * 100))

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
