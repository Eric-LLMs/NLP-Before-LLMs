# -*- coding: utf-8 -*-
import os,sys
import codecs
from gensim import corpora,models,similarities,utils
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import  nltk
from IntelligentAssistantWriting.conf.params import *

conf = create_params()
stop_words_file = os.path.join(conf.dir_root,'myworkspace/data/google_stopwords')

def filter_char(content):
    f = open(stop_words_file, "r")
    lines = f.readlines()
    s_words = []
    s_words.append('on.')
    s_words.append('.')
    for line in lines:
        s_words.append(line.replace('\r','').replace('\n',''))
    if content in s_words:
       return ''
    if ('.txt' in content) or (')' in content) or ('(' in content) or ('<' in content)or ('>' in content):
        content = ''
    else:
        content = content.replace(' ','').replace('\n', '').replace('.', '').replace(',', '').replace(';', '').replace('[', '').replace(']', '').replace(
            '{', '').replace('}', '').replace('(', '').replace(')', '').replace('!', '').replace('=', '').replace('?',
                                                                                                                  '').replace(
            '<<', '').replace('>>', '').replace('>', '').replace('<', '').replace('\'s', '').replace('"', '').replace('\\',
                                                                                                                      '').replace(
            ':', '').replace('+', '').replace('*', '').replace('&', '').replace('-', '').replace('$', '')
    return content

def get_tfidf_value(train_model_dir,query):
    # query.replace(',', '').replace('.','')
    # 载入字典
    dictionary = corpora.Dictionary.load(os.path.join(train_model_dir, "all.dic"))
    # 载入TFIDF模型和索引
    tfidfModel = models.TfidfModel.load(os.path.join(train_model_dir, "allTFIDF.mdl"))
    content = (query.lower()).split(' ')
    query_bow = dictionary.doc2bow(filter(lambda x: len(x) > 0, map(filter_char, content)))
    # 使用TFIDF模型向量化
    tfidfvect = tfidfModel[query_bow]
    list_result = {}
    for key,value in zip(filter(lambda x: len(x) > 0, map(filter_char, content)),tfidfvect):
        # print key,value[1]
        list_result[key] = float(value[1])
    # print list_result
    list_result = sorted(list_result.items(),key = lambda item:item[1],reverse=1)
    print('TF-IDF:')
    for w in list_result:
      print (w[0],w[1])

    # print dictionary.token2id

def get_textrank_keywords(query):
    # text = codecs.open('../test/doc/01.txt', 'r', 'utf-8').read()
    keywords_list = []
    text = query
    tr4w = TextRank4Keyword()

    tr4w.analyze(text=text, lower=True, window=3)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

    print('关键词：')
    for item in tr4w.get_keywords(20, word_min_len=1):
        print(item.word, item.weight)
        keywords_list.append(item)
    # print()
    # print('关键短语：')
    for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num=2):
        print(phrase)

    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')

    # print()
    print('摘要：')
    for item in tr4s.get_key_sentences(num=3):
        print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重

    return keywords_list

if __name__=='__main__':
    train_model_dir = os.path.join(conf.dir_root,'temp/train_temp')
    query = 'A healthy diet is very important for people . It contains some fat , some  fibre , a little salt and so on . The Chinese diet is considered to be the healthy  in the world . It contains a lot of fruit and vegetables . It is rich in fibre and low in sugar and fat . As a result , a lot of Chinese have healthy white teeth . So I am very happy to be Chinses . People in west eat too much fat and sugar . They like to eat butter , cream , chocolate , sweet and so on , which contain a lot of sugar . As a result , they put on weight easily . Many of them become very fat and some have bad teeth . The worset  thing is that many westerners die in an early age from the heart illness . So I thought   eat more fruit and vegetables , not to ear  too much sugar and fat is a real healthy diet .'
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(query)
    print (sentences)
    get_tfidf_value(train_model_dir,query)
    get_textrank_keywords(query)
    print (1)