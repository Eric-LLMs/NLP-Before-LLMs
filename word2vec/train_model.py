# -*- coding: utf-8 -*-

import logging
import sys
from gensim.models import word2vec
from pre_data import *
reload(sys)  # 重新载入sys
sys.setdefaultencoding("utf8")  # 设置默认编码格式

global model
dir_data = '/home/myworkspace/data/CORPUS_TXT'
def train():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # sentences = word2vec.Text8Corpus("E:/My_WorkSpace/Data/log/word2vec/data_w2v.txt")  # 加载语料
    sentences = get_sentences(dir_data)
    model = word2vec.Word2Vec(sentences, min_count=1, size=200)  # 训练skip-gram模型
    # model =word2vec.Word2Vec.load(u"E:/My_WorkSpace/Data/log/word2vec/geenlpke.model")
    # model = word2vec.Word2Vec.load_word2vec_format(u"E:/My_WorkSpace/Data/log/word2vec/test.model.bin", binary=True)

    # 计算两个词的相似度/相关程度
    y1 = model.similarity(u"book", u"is")
    print  y1
    print "--------\n"

    # 计算某个词的相关词列表
    y2 = model.most_similar(u"is", topn=100)  # 20个最相关的
    for item in y2:
        print item[0], item[1]
    print "--------\n"

    # 寻找对应关系
    print u"book  tree  is"
    y3 = model.most_similar([u'book', u'tree'], [u'is'], topn=3)
    for item in y3:
        print item[0], item[1]
    print "--------\n"

    # 寻找不合群的词
    y4 = model.doesnt_match(u"book  dog  is  number".split())
    print u"result：", y4
    print "--------\n"

    # 保存模型，以便重用
    model.save(u"./ltgword2vec.model")
    # 对应的加载方式
    # model_2 = word2vec.Word2Vec.load("text8.model")

    # 以一种C语言可以解析的形式存储词向量
    model.save_word2vec_format(u"./ltgword2vec.model.bin", binary=True)
    # 对应的加载方式
    # model_3 = word2vec.Word2Vec.load_word2vec_format("text.model.bin", binary=True)


if __name__ == "__main__":
    train()
