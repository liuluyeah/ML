from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import metrics
import tensorflow as tf
import numpy as np
import gensim
import pandas
import jieba

learn = tf.contrib.learn
MAX_DOCUMENT_LENGTH = 20
EMBEDDING_SIZE = 100  # cannot change, decided by word2ve model

def rnn_model(features,target):
  #change x_train into [20 2322 100] by using tf.unstack
  word_list=tf.unstack(features,axis=1) 

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

    # now change x_tr,x_te into x_train,x_test
    model=gensim.models.Word2Vec.load('wiki.zh.text100.model')
    # a is a padding vector
    a=np.array([ 0.01565487, -0.02834384,  0.0050781,  0.00281389, -0.01831529, -0.0281769,
	 -0.03484548, -0.03563496, -0.03443404, -0.00673436,  0.04364586, -0.01824741,
	 -0.01131345, -0.03640746,  0.00550056,  0.04395926,  0.0203858 , -0.00516302,
	  0.04008323, -0.0057266 ,  0.0365113 , -0.03460738, -0.01187438,  0.00549287,
	 -0.03652295, -0.00373745,  0.03293758,  0.00047863,  0.01827651, -0.03082007,
	 -0.03595891, -0.03405609,  0.0077058 ,  0.02505775, -0.04330837, -0.02034221,
	 -0.00177233, -0.01020397,  0.01452311, -0.03084193, -0.01693795,  0.04166659,
	  0.01166276,  0.00262658,  0.02123984, -0.00672483, -0.00728795, -0.00292311,
	 -0.03447647,  0.02672656, -0.01931424, -0.03280941,  0.03928034,  0.02634609,
	 -0.03776091, -0.00823639, -0.02328412,  0.02548924, -0.03731295, -0.0142525,
	 -0.0180848 ,  0.01957616, -0.02416089,  0.02907475, -0.01375897,  0.00587276,
	  0.03006439, -0.0306422 ,  0.0364229 ,  0.03659977,  0.00453732, -0.01905499,
	 -0.00101119,  0.02729544, -0.03571684,  0.03501784,  0.04202821, -0.02115477,
	  0.02117692,  0.03761742, -0.02283836, -0.00361463,  0.00468548,  0.0268892,
	 -0.03699357,  0.0433155 , -0.01639949, -0.01211649,  0.01257969, -0.01727303,
	 -0.01689328, -0.0313432 , -0.008336 ,   0.00480758, -0.02371452,  0.04339238,
	  0.04195571,  0.02062315, -0.0341436,   0.0405197 ])
    a_sentence=x_tr[0].strip().split(' ')
    x_tr_len=len(x_tr)
    firstword=a_sentence[0]
    try:
      st=model[firstword]
    except:
      st=a
    j=0
    for i in range(len(a_sentence)-1):
      j=j+1
      w=a_sentence[j]
      try:
        b=model[w]
      except:
        b=a
      st=np.vstack((st,b))
    while(len(st)<MAX_DOCUMENT_LENGTH):
      st=np.vstack((st,a)) 

    #fist [20,100] array has finished !
    # now construct  an arry [2322*20,100] 
    for one in x_tr[1:]:
      # consider the length of text <20 && >20
      one=one.strip().split(' ')
      if(len(one)<=MAX_DOCUMENT_LENGTH):
        firstword=one[0]
        try:
          nexts=model[firstword]
        except:
          nexts=a
        j=0
        # length=0 is special
        if((len(one)-1)==0):
          b=a
          nexts=np.vstack((nexts,b))
          while(len(nexts)<MAX_DOCUMENT_LENGTH):
            nexts=np.vstack((nexts,a))
        for i in range(len(one)-1):
          j=j+1
          w=one[j]
          try:
            b=model[w]
          except:
            b=a
          nexts=np.vstack((nexts,b))

        while(len(nexts)<MAX_DOCUMENT_LENGTH):
          nexts=np.vstack((nexts,a))
      # >20
      else:
        one=one[:MAX_DOCUMENT_LENGTH]
        firstword=one[0]
        try:
          nexts=model[firstword]
        except:
          nexts=a
        j=0
        for i in range(len(one)-1):
          j=j+1
          w=one[j]
          try:
            b=model[w]
          except:
            b=a
          nexts=np.vstack((nexts,b))
      st=np.vstack((st,nexts))
    x_train=st.reshape((x_tr_len,MAX_DOCUMENT_LENGTH,100))
    x_train=x_train.astype(np.float32)
    # now construct test array
    x_te_len=len(x_te)
    te_sentence=x_te[0].strip().split(' ')
    teword=te_sentence[0]
    try:
      st=model[teword]
    except:
      st=a
    j=0
    for i in range(len(te_sentence)-1):
      j=j+1
      w=te_sentence[j]
      try:
        b=model[w]
      except:
        b=a
      st=np.vstack((st,b))
    while(len(st)<MAX_DOCUMENT_LENGTH):
      st=np.vstack((st,a)) #fist [20,100] array has finished !

    # now construct  an arry [2322*20,100] 
    for one in x_te[1:]:
      # consider the length of text <20 && >20
      one=one.strip().split(' ')
      if(len(one)<=MAX_DOCUMENT_LENGTH):
        firstword=one[0]
        try:
          nexts=model[firstword]
        except:
          nexts=a
        j=0
        # length=0 is special
        if((len(one)-1)==0):
          b=a
          nexts=np.vstack((nexts,b))
          while(len(nexts)<MAX_DOCUMENT_LENGTH):
            nexts=np.vstack((nexts,a))
        for i in range(len(one)-1):
          j=j+1
          w=one[j]
          try:
            b=model[w]
          except:
            b=a
          nexts=np.vstack((nexts,b))

        while(len(nexts)<MAX_DOCUMENT_LENGTH):
          nexts=np.vstack((nexts,a))
      # >20
      else:
        one=one[:MAX_DOCUMENT_LENGTH]
        firstword=one[0]
        try:
          nexts=model[firstword]
        except:
          nexts=a
        j=0
        for i in range(len(one)-1):
          j=j+1
          w=one[j]
          try:
            b=model[w]
          except:
            b=a
          nexts=np.vstack((nexts,b))
      st=np.vstack((st,nexts))
    x_test=st.reshape((x_te_len,MAX_DOCUMENT_LENGTH,100))
    x_test=x_test.astype(np.float32)

    return(x_train,y_tr,x_test,y_te)

def main():
  x_train,y_train,x_test,y_test = prepare_data()
  # Build model
  model_fn = rnn_model
  classifier = learn.Estimator(model_fn=model_fn)
  # Train and predict
  classifier.fit(x=x_train,y=y_train,steps=2000,batch_size=200)
  y_predicted = [
      p['class'] for p in classifier.predict(
          x_test, as_iterable=True)
  ]
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy: {0:f}'.format(score))
  print(y_predicted)

if __name__ == '__main__':
  main()

              







