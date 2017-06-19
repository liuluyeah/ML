#!/usr/bin/python
#-*-coding:utf-8-*-
import os
import jieba
import nltk

def prepareData():
    train_path='/data/liulu/SMP2017-ecdt/data/train'
    test_path='/data/liulu/SMP2017-ecdt/data/develop'
    train_files=os.listdir(train_path) #31
    test_files=os.listdir(test_path)
    train_set=[]
    test_set=[]
    all_words={}
    #prepare train data
    i=0
    for file in train_files:
        with open(train_path+'//'+file,'r') as f:
            for line in f.readlines():
                word_cut=jieba.cut(line,cut_all=False)
                word_list=list(word_cut)
                for word in word_list:
                    if word in all_words.keys():
                        all_words[word]+=1
                    else:
                        all_words[word]=0
                train_set.append((word_list,i))
        i=i+1
    print(len(train_set))
    #prepare test data
    j=0
    for file in test_files:
        with open(test_path+'//'+file,'r') as f:
            for line in f.readlines():
                word_cut=jieba.cut(line,cut_all=False)
                word_list=list(word_cut)
                for word in word_list:
                    if word in all_words.keys():
                        all_words[word]+=1
                    else:
                        all_words[word]=0
                test_set.append((word_list,j))
        j=j+1

    all_words_list=sorted(all_words.items(),key=lambda e:e[1],reverse=True)
    word_features=[]
    print('all words number is: %d'%len(all_words_list))
	# construct word features
    for t in range(len(all_words_list)):
        word_features.append(all_words_list[t][0])
		
	return train_set,test_set,word_features

def document_features(document,word_features):
    document_words=set(document)
    features={}
    for word in word_features:
        features['contains(%s)'%word]=(word in document_words)
    return(features)
	
if __name__=='__main__':
    train_set,test_set,word_features=prepareData()
    train_data=[(document_features(d,word_features),c) for (d,c) in train_set]
    test_data=[(document_features(d,word_features),c) for (d,c) in test_set]
    print("train number:%d"%len(train_data))
    print("test number:%d"%len(test_data))
    classifier=nltk.NaiveBayesClassifier.train(train_data)
    print(nltk.classify.accuracy(classifier,test_data))
