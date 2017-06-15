#coding=utf8
import numpy, jieba
from pymongo import MongoClient
import codecs, time, sys
from pyltp import Segmentor
from sentences_trunk import get_sentence_trunk, get_structure_words

# sys.path.append('../')
from src.common import utils

def cos_calculate(x, y):
    '''计算两个向量的余弦夹角'''

    x = numpy.array(x)
    y = numpy.array(y)
    len_x = numpy.sqrt(x.dot(x))
    len_y = numpy.sqrt(y.dot(y))
    return x.dot(y) / (len_x * len_y)


def get_my_sentence_vector(sentence_words):
    '''计算句子向量'''

    result = [0.0 for x in xrange(200)]

    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['words']
    N = 0
    for w in sentence_words:
        item = coll.find_one({'content': w, 'vector': {'$exists': True}})
        if item:
            N += 1
            vector = item['vector']
            for i in xrange(200):
                result[i] += vector[i]
    mc.close()

    if N > 0:
        result = [x / N for x in result]
    
    return result


def get_vector_similarity_top_n(sentence):
    '''返回与句子库中向量相似度最高的若干句子'''

    # 句法分析，求句子结构
    my_sentence_parsing = utils.sentence_parsing(sentence)
    my_sentence_trunk = get_sentence_trunk(my_sentence_parsing)
    my_key_words = get_structure_words(my_sentence_trunk, my_sentence_parsing)
    my_key_words = [x for x in my_key_words if x[0] in ['HED']]

    # 分词，求句子向量
    sentence_words = utils.word_segmentation_by_sentence(sentence)
    sentence_words = [x.decode('utf8') for x in sentence_words]
    my_sentence_vector = get_my_sentence_vector(sentence_words)

    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['sentences_new']

    result = []
    # 找出所有具有句向量的句子，且对句子长度进行了限制
    items = coll.find({'vector': {'$exists': True},
                'length': {'$gt': len(sentence) * 0.5, '$lt': len(sentence) * 2}})
    for item in items[:]:
        if item.has_key('structure_words'):
            key_words_A = item['structure_words']
            key_words_A = [x for x in key_words_A if x[0] in ['HED']]

            # 句子主干词语保持一致
            if ''.join([x[1] for x in my_key_words]) == ''.join([x[1] for x in key_words_A]):
                vector_A = item['vector']
                similarity = cos_calculate(vector_A, my_sentence_vector)
                if similarity > 0.5:
                    # print similarity, item['content']
                    result.append((similarity, item['content']))

    mc.close()

    result.sort(reverse=True)
    if len(result) >= 5:
        result = [x[1] for x in result[:5]]
    else:
        result = [x[1] for x in result]

    # for x in result:
    #     print x
    
    return result


def get_sentence_vector(sentence_words, vectors):
    '''计算句子向量'''

    result = [0.0 for x in xrange(200)]
    
    N = 0
    for w in sentence_words:
        if vectors.has_key(w):
            N += 1
            for i in xrange(200):
                result[i] += vectors[w][i]

    ratio = N * 1.0 / len(sentence_words)
    if ratio > 0.8:
        # 80% 以上的词语具有词向量，才返回该句子向量
        # 否则可认为句子低频词过多，从而句子也为低频句
        result = [x / N for x in result]
        return result
    else:
        # 低频句，则该句子没有词向量
        return None


def update_sentences_vector():
    '''给句子库添加句向量字段'''


    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['sentences']

    # 从 vectors.bin 读入词向量模型，得到词向量矩阵
    in_file = codecs.open('../words/vectors.bin', 'r', encoding='utf8')
    lines = [x.strip() for x in in_file.readlines()]
    in_file.close()

    vectors = {}
    for line in lines[1:]:
        temp_list = line.split()
        key = temp_list[0]
        value = [float(x) for x in temp_list[1:]]
        vectors[key] = value

    num = 0
    items = coll.find({'trunk_depth_3': {'$exists': True}})
    for item in items[:]:
        if not item.has_key('vector'):
            sentence_words = [x['cont'] for x in item['parsing']]
            sentence_vector = get_sentence_vector(sentence_words, vectors)
            if sentence_vector is not None:
                # print sentence_vector
                # print len(sentence_vector)
                coll.update({'content': item['content']}, {'$set': {'vector': sentence_vector}})
            else:
                print item['content']
                coll.remove({'content': item['content']})

        num += 1
        if num % 10000 == 0:
            print num

    mc.close()


if __name__ == '__main__':
    start_time = time.time()
    #############

    get_vector_similarity_top_n(u'快乐是付出艰辛努力时手捧奖杯的泪水。')
    # get_vector_similarity_top_n(u'亲人的爱就像阳光，总是那么温暖，那么默默无闻，细心地呵护我的成长！')
    # update_sentences_vector()

    #############
    end_time = time.time()
    print 'Finished in %.2f secondes:' % (end_time - start_time)
