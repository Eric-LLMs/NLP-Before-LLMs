#coding=utf8
from pymongo import MongoClient
import re, json, time, random,sys
import requests
import LTP_sentence_split
from pyltp import Parser

# sys.path.append('../')
# from common import utils


def split_sentences():
    '''将作文进行句子切分，建立句子库，同时进行句法分析'''

    mc = MongoClient('127.0.0.1')
    coll_source = mc['zuowen']['articles']
    coll_target = mc['zuowen']['sentences']
    coll_target.ensure_index('content', unique=True)    # 创建唯一索引

    parser = Parser()
    parser.load('/Documents/Study/Python/3.3.0/ltp_data/parser.model')

    num = 0
    items = coll_source.find({'pos_tags': {'$exists': True}})    # 所有词性标注后的作文
    for item in items[:1]:
        sentence_postags = item['pos_tags']
        for sentence_postag in sentence_postags:
            word_pos = [xx.split('_') for xx in sentence_postag.split('||')]
            words = [x[0] for x in word_pos]
            postags = [x[1] for x in word_pos]
            sentence = ''.join(words)

            # 对句子长度进行控制(5 - 60) AND 句库中不存在该句子
            if (len(sentence) >= 5 and len(sentence) <= 50) and \
                (not coll_target.find_one({'content': sentence})):
                # 编码转换
                try:
                    words = [x.encode('utf8') for x in words]
                    postags = [x.encode('utf8') for x in postags]
                    arcs = parser.parse(words, postags)

                    resultJSON = []
                    for index in xrange(len(words)):
                        resultJSON.append({'id': index, 
                                            'cont': words[index], 
                                            'pos': postags[index], 
                                            'parent': arcs[index].head - 1, 
                                            'relate': arcs[index].relation
                                            })

                    # print resultJSON
                    dic = {}
                    dic['content'] = sentence
                    dic['parsing'] = resultJSON
                    coll_target.save(dic)

                except Exception, e:
                    print str(e)
                    print len(sentence), sentence
                    print len(words), words
                    print len(postags), postags
                    print len(arcs), arcs
                    print
                    continue  

        num += 1
        if num % 1000 == 0:
            print num
    
    parser.release()
    mc.close()


def copy_sentences_to_new_collection():
    '''备份句子库'''

    mc = MongoClient('127.0.0.1')
    coll_source = mc['zuowen']['sentences']
    coll_target = mc['zuowen']['sentences_new']
    coll_target.ensure_index('content', unique=True)    # 创建唯一索引

    num = 0
    items = coll_source.find({'trunk_depth_3': {'$exists': True}})
    for item in items[500000:1000000]:
        coll_target.save(item)

        num += 1
        if num % 10000 == 0:
            print num

    mc.close()


if __name__ == '__main__':
    start_time = time.time()
    #############

    # split_sentences()
    # copy_sentences_to_new_collection()

    #############
    end_time = time.time()
    print 'Finished in %.2f secondes:' % (end_time - start_time)
