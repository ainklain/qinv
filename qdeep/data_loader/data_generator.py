import numpy as np
import pickle
import tensorflow as tf
#
# class DataGenerator:
#     def __init__(self, config):
#         self.config = config
#         # load data here
#         self.input = np.ones((500, 784))
#         self.y = np.ones((500, 10))
#
#     def next_batch(self, batch_size):
#         idx = np.random.choice(500, batch_size)
#         yield self.input[idx], self.y[idx]


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.make_dataset(config.data_file_dir)

    def make_dataset(self, file_dir):
        # file_dir = 'D:\\DeepInvest\\Code\\DeepInv\\data\\fundamentals.pkl'
        f = open(file_dir, 'rb')
        data_pickle = pickle.load(f)
        data_ = data_pickle['data']
        label_ = data_pickle['label']

        n_input = data_.shape[1]
        n_output = label_.shape[1]

        assert self.config.n_input == n_input, 'config.n_input is not same as that of data_file'
        assert self.config.n_output == n_output, 'config.n_output is not same as that of data_file'


        with tf.variable_scope('data'):

            train_split = int(len(data_) * 0.8)
            # train_split = 1000
            train_data = tf.data.Dataset.from_tensor_slices(data_[:train_split])
            train_label = tf.data.Dataset.from_tensor_slices(label_[:train_split])
                # .map(lambda z: tf.one_hot(tf.cast(z, tf.int32), 2))
            train_dataset = tf.data.Dataset.zip((train_data, train_label)).shuffle(10000).repeat().batch(self.config.batch_size)

            valid_data = tf.data.Dataset.from_tensor_slices(data_[train_split:])
            valid_label = tf.data.Dataset.from_tensor_slices(label_[train_split:])
                # .map(lambda z: tf.one_hot(tf.cast(z, tf.int32), 2))
            valid_dataset = tf.data.Dataset.zip((valid_data, valid_label)).shuffle(10000).repeat().batch(self.config.batch_size)

            x_test = tf.placeholder(tf.float64, [None, n_input])
            y_test = tf.placeholder(tf.float64, [None, n_output])
            test_data = tf.data.Dataset.from_tensor_slices(x_test)
            test_label = tf.data.Dataset.from_tensor_slices(y_test)
            test_dataset = tf.data.Dataset.zip((test_data, test_label)).batch(1)

            x_gen = tf.placeholder(tf.float64, [None, n_input])
            x_data = tf.contrib.data.

            # iterator = train_dataset.make_initializable_iterator()
            iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

            self.n_train = len(data_[:train_split])
            self.n_valid = len(data_[train_split:])
            self.n_test = 1

            self.next_batch = iterator.get_next()
            self.train_init_op = iterator.make_initializer(train_dataset, 'train_init_op')
            self.valid_init_op = iterator.make_initializer(valid_dataset, 'valid_init_op')
            self.test_init_op = iterator.make_initializer(test_dataset, 'test_init_op')

            tf.add_to_collection('train_init_op', self.train_init_op)
            tf.add_to_collection('valid_init_op', self.valid_init_op)
            tf.add_to_collection('test_init_op', self.test_init_op)



            # inputs, labels = iterator.get_next()
            #
            # train_init_op = iterator.make_initializer(train_dataset)
            # valid_init_op = iterator.make_initializer(valid_dataset)


