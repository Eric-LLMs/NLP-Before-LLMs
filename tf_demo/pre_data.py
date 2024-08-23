#-*- coding: utf-8 -*-
import os,sys
import numpy as np
import json
import codecs
import h5py
import base64
from bs4 import re
from sklearn.cross_validation import train_test_split

corpus_tags = ['2','1', '3', '4']  #tag_4
maxlen = 200
filder_chars = ['\n','\r']

fd = open(os.path.join(sys.path[0],'./stop_words'), 'r')
chars = fd.readlines()
for char in chars :
   filder_chars.append(char[:-1])

def load_data(data_path):
       print 'train from file', data_path
       delims = [' ', '\n']
       vocab, indexVocab = generate_vocab(data_path)
       X, y, initProb = doc2vec(data_path, vocab)
       print len(X), len(y), len(vocab), len(indexVocab)
       print initProb
       return (X, y), initProb, (vocab, indexVocab)

def sent2vec(sent, vocab, ctxWindows=5):
       chars = []
       for char in sent:
           chars.append(char)
       return sent2vec2(chars, vocab, ctxWindows=ctxWindows)

def sent2vec2(sent, vocab, ctxWindows=5):
       charVec = []
       for char in sent:
           if char in vocab:
               charVec.append(vocab[char])
           else:
               charVec.append(vocab['unknown_w'])

       # add padding (head and end)
       # 不足补齐
       num = len(charVec)
       if num < ctxWindows:
           pad = (ctxWindows - num) / 2
           for i in range(pad):
               charVec.insert(0, vocab['padd_w'])
               charVec.append(vocab['padd_w'])
           if len(charVec) == ctxWindows - 1:
               charVec.append(vocab['padd_w'])

       X = []
       # 截取ctxWindows 的长度
       for i in range(ctxWindows):
           X.append(charVec[i])
       return X

def doc2vec(fname, vocab):
       '''文档转向量'''

       # 一次性读入文件，注意内存
       import sys
       reload(sys)
       sys.setdefaultencoding('utf8')
       type = sys.getfilesystemencoding()
       # 一次性读入文件，注意内存
       fd = codecs.open(fname, 'r', 'utf-8')
       lines = fd.readlines()
       fd.close()

       # 样本集
       X = []
       y = []

       # 标注统计信息
       tagSize = len(corpus_tags)
       tagCnt = [0 for i in range(tagSize)]

       # 遍历行
       for line in lines:
           # 按空格分割
           items = line.split('\t')
           tag = items[1].replace('\r', '').replace('\n', '')
           content = items[0]
           content = re.sub("[A-Za-z0-9]", "", content)
           words = content
           # 每行的分词信息
           chars = []
           # 遍历字
           for word in words:

               if word in filder_chars:
                   continue
               chars.append(word)

           # 字向量表示
           lineVecX = sent2vec2(chars, vocab, ctxWindows=maxlen)

           # 统计标注信息
           lineVecY = corpus_tags.index(tag)

           tagCnt[lineVecY] += 1

           X.append(lineVecX)
           y.append(lineVecY)

       # 文章总频次
       docCnt = sum(tagCnt)
       # tag初始概率
       initProb = []
       for i in range(tagSize):
           initProb.append(tagCnt[i] / float(docCnt))

       return X, y, initProb

def savePreInfo(path, cwsInfo):
       '''save Stm info '''
       print('save Stm info to %s' % path)
       fd = open(path, 'w')
       initProb, (vocab, indexVocab) = cwsInfo
       j = json.dumps(initProb)
       fd.write(j + '\n')
       for char in vocab:
           fd.write(char.encode('utf-8') + '\t' + str(vocab[char]) + '\n')
       fd.close()

def savePreData(path, cwsData):
       '''save Stm data'''
       print('save Stm data to %s' % path)
       # h5py to save
       fd = h5py.File(path, 'w')
       (X, y) = cwsData
       fd.create_dataset('X', data=X)
       fd.create_dataset('y', data=y)
       fd.close()

def generate_vocab(file_name, delimiters=[' ', '\n']):
       import sys
       reload(sys)
       sys.setdefaultencoding('utf8')
       type = sys.getfilesystemencoding()
       # 一次性读入文件，注意内存
       fd = codecs.open(file_name, 'r', 'utf-8')
       data = fd.read()
       fd.close()
       vocab = {}
       indexVocab = []
       # 遍历
       index = 1
       data = re.sub('[A-Za-z0-9]', '', data)
       for char in data:
           if char in filder_chars:
               continue
           if char in vocab:
               continue
           try:
               print char
               if char not in delimiters:
                   vocab[char] = index
                   indexVocab.append(char)
                   index += 1
           except Exception as Error:
               print('genVocab Error')

       # 加入未登陆新词和填充词
       # vocab['unknown_w'] = len(vocab)
       vocab['unknown_w'] = 0
       # 用0补齐
       vocab['padd_w'] = 0
       indexVocab.append('unknown_w')
       indexVocab.append('padd_w')
       # 返回字典与索引
       return vocab, indexVocab

def loadPreData(path):
       '''load pre data'''
       print('load pre data from %s' % path)
       fd = h5py.File(path, 'r')
       X = fd['X'][:]
       y = fd['y'][:]
       fd.close()
       return (X, y)

def loadPreInfo(path):
       '''load pre info'''
       print('load pre info from %s' % path)
       fd = open(path, 'r')
       line = fd.readline()
       j = json.loads(line.strip())
       initProb = j
       lines = fd.readlines()
       fd.close()
       vocab = {}
       indexVocab = [0 for i in range(len(lines))]
       for line in lines:
           rst = line.strip().split('\t')
           if len(rst) < 2: continue
           char, index = rst[0].decode('utf-8'), int(rst[1])
           vocab[char] = index
           indexVocab[index] = char
       return initProb, (vocab, indexVocab)

def pre_data(data_path, pre_data_path, pre_info_path):
       # 文本向量转换
       (X, y), initProb, (vocab, indexVocab) = load_data(data_path)
       # 保存向量信息
       StmInfo = initProb, (vocab, indexVocab)
       StmData = (X, y)
       savePreInfo(pre_info_path, StmInfo)
       savePreData(pre_data_path, StmData)
       return pre_data_path, pre_info_path


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes

    # Returns
        A binary matrix representation of the input.
    '''
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def data_to_vector(preInfo, preData, size):
    maxlen = 200
    batch_size = 128
    initProb, (vocab, indexVocab) = preInfo
    (X, y) = preData

    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=size, random_state=1)
    max_features = len(indexVocab)
    print ('max_features indexVocab')
    print max_features

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    return (train_X, train_y), (test_X, test_y)

    # nb_classes = len(corpus_tags)
    # if not train_size:
    #     train_size = 0.9

    # train_y = train_y[:, np.newaxis]
    # test_y = test_y[:, np.newaxis]

    # print ('X_train :')
    # print  train_X.shape
    #
    # print ('X_test :')
    # print  test_X.shape
    #
    #
    # Y_train = to_categorical(train_y, nb_classes)
    # Y_test = to_categorical(test_y, nb_classes)
    #
    # print ('Y_train_toC :')
    # print  Y_train.shape
    #
    # print ('Y_test_toC :')
    # print  Y_test.shape
    #
    # print ('Y_train :')
    # print  train_y.shape
    #
    # print ('Y_test :')
    # print  test_y.shape



def convert_to_vec_demo(log_dir, data_file_name):
       data_path = os.path.join(log_dir, data_file_name)
       pre_info_path = os.path.join(log_dir, 'Pre.info')
       pre_data_path = os.path.join(log_dir, 'Pre.data')
       train_size = 0.9
       # model_path = os.path.join(log_dir, 'Pre.model')
       # weight_path = os.path.join(log_dir, 'Pre.model.weights')
       try:
           pre_data(data_path, pre_data_path, pre_info_path)
       except Exception as Error:
           print 'Error'
       try:
           # 加载保存的数据
           preInfo = loadPreInfo(pre_info_path)
           preData = loadPreData(pre_data_path)
           # 训练并保存模型
           # result = stm_lstm_train(preInfo, preData, model_path, weight_path)
           (train_X, train_y), (test_X, test_y) = data_to_vector(preInfo, preData, train_size)
           print ('ok')
       except Exception as Error:
           print 'Error'

