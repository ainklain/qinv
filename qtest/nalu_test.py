import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

import os
from qtest.nalu import NALU

### dataset
seed = 1234
num_samples = 1000
mode='interpolation'
op_fn = None

task_name = 'square'
task_fn = lambda x, y: x * y
op_fn = task_fn

np.random.seed(0)

static_index = np.arange(0, 100, dtype=np.int64)
np.random.shuffle(static_index)

np.random.seed(seed)  # make deterministic

print("Generating dataset")

# Get the input stream
X = np.random.rand(num_samples, 100)

if mode == 'extrapolation':
    X *= 2.

# Select the slices on which we will perform the operation
a_index, b_index = static_index[:(len(static_index) // 2)], static_index[(len(static_index) // 2):]
a = X[:, a_index]
b = X[:, b_index]

# Get the sum of the slices
a = np.sum(a, axis=-1, keepdims=True)
b = np.sum(b, axis=-1, keepdims=True)

# perform the operation on the slices in order to get the target
Y = op_fn(a, b)


units = 2

ip = Input(shape=(100, ))
x = NALU(units)(ip)
x = NALU(1)(x)

model = Model(ip, x)
model.summary()

optimizer = RMSprop(0.1)
