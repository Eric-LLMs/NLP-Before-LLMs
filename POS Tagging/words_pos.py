#coding=utf8
import time, sys
from pymongo import MongoClient
from pyltp import Postagger
from collections import defaultdict

# sys.path.append('../')
from src.common import utils


def word_pos_tagging_LTP():
    '''从数据库中读取作文分词结果，进行词性分析'''

    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['articles']

    num = 0
    postagger = Postagger()
    postagger.load('/Users/Documents/Study/Python/3.3.0/ltp_data/pos.model')

    items = coll.find({'word_segment': {'$exists': True}})    # 所有分词后的作文
    for item in items[:100000]:
        if not item.has_key('pos_tags'):
            try:
                sentences = item['word_segment']
                postag_list = []
                for words in sentences:
                    words = [x.encode('utf8') for x in words]
                    postags = postagger.postag(words)
                    word_pos_tuples = ['%s_%s' % (words[i], postags[i]) for i in xrange(len(words))]
                    postag_list.append('||'.join(word_pos_tuples))

            except Exception, e:
                print str(e)
                continue
            
            coll.update({'id_url': item['id_url']}, {'$set': {'pos_tags': postag_list}})

            num += 1
            if num % 1000 == 0:
                print num

    postagger.release()
    mc.close()


def default_value():
    return defaultdict(int)


def update_word_pos_mongo():
    '''给词库添加词性 pos 字段'''

    mc = MongoClient('127.0.0.1')
    coll_source = mc['zuowen']['articles']
    coll_target = mc['zuowen']['words']

    word_pos_dict = defaultdict(default_value)
  
    # 开始统计
    num = 0
    items = coll_source.find({'pos_tags': {'$exists': True}})    # 所有词性标注后的作文
    for item in items:
        sentences = item['pos_tags']
        for sentence in sentences:
            for word_pos in [xx.split('_') for xx in sentence.split('||')]:
                try:
                    if len(word_pos) == 2:
                        word_pos_dict[word_pos[0]][word_pos[1]] += 1
                        # print word_pos[0], word_pos[1]

                except Exception, e:
                    print sentence
                    print str(e)

        num += 1
        if num % 10000 == 0:
            print num

    # 开始保存
    num = 0
    items = coll_target.find()
    for item in items:
        try:
            word = item['content']
            if word_pos_dict.has_key(word):
                coll_target.update({'content': word}, {'$set': {'pos': word_pos_dict[word]}})
            else:
                coll_target.update({'content': word}, {'$set': {'pos': {}}})
        except Exception, e:
            print word
            print str(e)

        num += 1
        if num % 10000 == 0:
            print num

    # for k, v in word_pos_dict.items():
    #     print k,
    #     for x, y in v.items():
    #         print x, y

    mc.close()


def computer_pos_probability():
    '''在词库中，计算词性频率分布信息，并添加字段'''

    AA = set(['a', 'b'])
    NN = set(['n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz'])

    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['words']

    num = 0
    items = coll.find()
    for item in items[:]:
        pos_dic = item['pos']
        ALL = sum([v for v in pos_dic.values()])
        
        dic = {}
        AA_n, NN_n = 0, 0
        for k, v in pos_dic.items():
            dic[k] = v * 1.0 / ALL
            if k in AA:
                AA_n += v
            elif k in NN:
                NN_n += v
            
        # 合并词性
        if AA_n > 0:
            dic['AA'] = AA_n * 1.0 / ALL
        if NN_n > 0:
            dic['NN'] = NN_n * 1.0 / ALL

        coll.update({'content': item['content']}, {'$set': {'pos_info': dic}})

        num += 1
        if num % 10000 == 0:
            print num

    mc.close()


if __name__ == '__main__':
    start_time = time.time()
    #############

    # word_pos_tagging_LTP()
    # update_word_pos_mongo()
    # computer_pos_probability()

    #############
    end_time = time.time()
    print 'Finished in %.2f secondes:' % (end_time - start_time)
