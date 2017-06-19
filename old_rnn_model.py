from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.layers.python.layers import encoders
from sklearn import metrics
import tensorflow as tf
import numpy as np
import pandas
import jieba
import sys

learn = tf.contrib.learn
FLAGS = None
MAX_DOCUMENT_LENGTH = 20
EMBEDDING_SIZE = 100
n_words=0

def rnn_model(features,target):
  """RNN model to predict from sequence of words to a2 class."""
  word_vectors=tf.contrib.layers.embed_sequence(features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE, scope='words')
  word_list=tf.unstack(word_vectors,axis=1)
  # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
  # cell = tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)
  cell = tf.contrib.rnn.BasicLSTMCell(EMBEDDING_SIZE,state_is_tuple=False)
  lstm_cell = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=1.0,output_keep_prob=0.5,seed=None)
  # Create an unrolled Recurrent Neural Networks to length of
  # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
  _, encoding = tf.contrib.rnn.static_rnn(lstm_cell, word_list,dtype=tf.float32)
  target = tf.one_hot(target, 31, 1, 0)
  logits = tf.contrib.layers.fully_connected(encoding, 31, activation_fn=None)
  #loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
  loss=tf.contrib.losses.mean_squared_error(logits, target)
  # Create a training op.
  train_op = tf.contrib.layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='Adam',
      learning_rate=0.01)

  return ({
      'class': tf.argmax(logits, 1),
      'prob': tf.nn.softmax(logits)
  }, loss, train_op)

def prepare_data():
    train_x_list=[]
    train_y_list=[]
    test_x_list=[]
    test_y_list=[]
    # convert 31 files into one Series format as x_tr,y_tr
    for i in xrange(31):
      file_to_list=[]
      text_to_list=[]
      with open("/data/liulu/SMP2017-ecdt/data/train/"+str(i)+".txt","r") as fd:
          for line in fd.readlines():
              file_to_list.append(list(map(str,line.split())))
      for item in file_to_list:
          for it in item:
              words = jieba.cut(it)
              textstr=""
              for w in words:
                  textstr+=w+' '
              text_to_list.append(textstr)

      onelabel=[]
      for item in xrange(len(text_to_list)):
          onelabel.append(i)
      train_y_list.append(pandas.Series(onelabel))
      train_x_list.append(pandas.Series(text_to_list))
    x_tr = pandas.concat(train_x_list)
    y_tr = pandas.concat(train_y_list)
    x_tr = x_tr.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)

    # here is the test data
    for i in xrange(31):
      file_list=[]
      text_list=[]
      with open("/data/liulu/SMP2017-ecdt/data/develop/"+str(i)+".txt","r") as fd:
          for line in fd.readlines():
              file_list.append(list(map(str,line.split())))
      for item in file_list:
          for it in item:
              words = jieba.cut(it,cut_all=False)
              textstr=""
              for w in words:
                  textstr+=w+' '
              text_list.append(textstr)
      onelabel=[]
      for item in xrange(len(text_list)):
          onelabel.append(i)
      test_y_list.append(pandas.Series(onelabel))
      test_x_list.append(pandas.Series(text_list))
    x_te = pandas.concat(test_x_list)
    y_te = pandas.concat(test_y_list)
    x_te = x_te.reset_index(drop=True)
    y_te = y_te.reset_index(drop=True)

    return(x_tr,y_tr,x_te,y_te)

def main():
  global n_words
  x_train,y_train,x_test,y_test = prepare_data()
  print(x_train[0][0])

  # Process vocabulary
  vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)
  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))
  n_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % n_words)

  # Build model
  model_fn = rnn_model
  classifier = learn.Estimator(model_fn=model_fn)

  # Train and predict
  classifier.fit(x=x_train,y=y_train,steps=1000,batch_size=100)
  y_predicted = [
      p['class'] for p in classifier.predict(
          x_test, as_iterable=True)
  ]
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy: {0:f}'.format(score))
  print(y_predicted)

if __name__ == '__main__':
  main()


