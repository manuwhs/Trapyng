# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import tensorflow as tf

Py3 = sys.version_info[0] == 3


"""
Description: 
"""
def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  
  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, word_to_id, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
     
    """
    Our final goal if first do structure the data in batches, where every batch
    a set of "batch_size" chains, each chain with "num_step" samples with D dimension
    The final structure of a batch is:
        Batch_data[batch_size, num_steps, D]
    The number of samples in every batch will be batch_len = num_steps x D
    For this we specify the length of the chains an the number of chains per batch.
    We later compute how many batches we have given the data.
    """
    # We first convert the list of words to a tensor...
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    # Compute the number of batches given the batch size
    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    
    # Divide the data in the different batches. Reshape the data as [batch_size, batch_len] 
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    # Compute the numbet of Batches we can generate.
    epoch_size = (batch_len - 1) // num_steps
    
    # Check that we have at least one batch and create a variable with number of batches.
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")

    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    ## Create the index of the batches "i" which will change every time is called in the session.
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    
    """
    Now we use this index to select and return the data corresponding to the i-th batch.
    Where i-th will change every time during the session.
    """
    
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    
    print (["X batch shape", x.shape])
    return x, y


def Artificial_data_producer(X, Y, batch_size, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  
  with tf.name_scope(name, "PTBProducer", [X,Y, batch_size]):
     
    """
    Our final goal if first do structure the data in batches, where every batch
    a set of "batch_size" chains, each chain with "num_step" samples with D dimension
    The final structure of a batch is:
        Batch_data[batch_size, num_steps, D]
    The number of samples in every batch will be batch_len = num_steps x D
    For this we specify the length of the chains an the number of chains per batch.
    We later compute how many batches we have given the data.
    """

    print ("------------- Slicing the data DATAPRODUCER ------------")
#    print ([len(X),len(Y)])
    X = np.stack(X, axis = 0)
    Y = np.stack(Y, axis = 0)
    
    print ("Shape of concatenated X",X.shape)
    num_chains, num_steps, X_dim = X.shape
    Y = Y.reshape((num_chains, num_steps))
    # We first convert the list of words to a tensor...
    X = tf.convert_to_tensor(X, name="raw_data_X", dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, name="raw_data_Y", dtype=tf.int32)
    
    # Compute the number of batches given the batch size
#    data_len = tf.size(Y)
    batch_len = num_steps * batch_size;
    
    
    # Compute the numbet of Batches we can generate.
    epoch_size = int(num_chains / batch_size)
    
    # Divide the data in the different batches. Reshape the data as [batch_size, batch_len] 
    print ("Initial number of chains = %i"%num_chains)
    print ("Number of steps in a chain: step_size = %i"%num_steps)
    print ("Number of dimensions of a sample: X_dim = %i"%X_dim)
    print ("Number of chains per batch: batch_size = %i"%batch_size)
    print ("Number of samples in a batch: batch_len = batch_size x num_steps = %i"%batch_len)

    print ("Number of batches created: epoch_size = %i"%epoch_size)
    print ("Last %i chains not used since not enough for batch"%(num_chains - epoch_size*batch_size))
    
    print (["Shape of X converted to tensor [total_num_chains, step_num, Xdim]",X.shape])
    print (["Shape of Y converted to tensor [total_num_chains, step_num, Xdim]",Y.shape])
    
    # We now remove only take as many chains as they fin in the batchs
    Xdata = tf.reshape(X[0 :epoch_size*batch_size],
                      [epoch_size*batch_size, num_steps,X_dim])
    
    print (["Shape of Xdata reshapes converted to tensor",Xdata.shape]) 
    Ydata = tf.reshape(Y[0:epoch_size*batch_size],
                      [epoch_size*batch_size, num_steps])

    print (["Shape of Ydata reshapes converted to tensor",Ydata.shape])  
    # Check that we have at least one batch and create a variable with number of batches.
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")

    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    ## Create the index of the batches "i" which will change every time is called in the session.
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    
    """
    Now we use this index to select and return the data corresponding to the i-th batch.
    Where i-th will change every time during the session.
    """
    
    x = tf.strided_slice(Xdata, [i* batch_size,0,0],
                                [(i+1)* batch_size, num_steps, X_dim])
    
    y = tf.strided_slice(Ydata, [i* batch_size,0],
                                [(i+1)* batch_size, num_steps])
#    
#    x = Xdata[ i * batch_size:  (i+1) * batch_size,:,:]
#    y = Ydata[ i * batch_size:  (i+1) * batch_size,:]
    
    x.set_shape([batch_size, num_steps, X_dim])
    y.set_shape([batch_size, num_steps])
    
    print (["X batch shape", x.shape])
    print (["Y batch shape", y.shape])
    return x, y
