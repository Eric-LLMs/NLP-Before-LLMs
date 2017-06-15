#coding=utf8
from pymongo import MongoClient
import sentences_vector, sentences_trunk
import time, random


def get_similar_sentence_top_n(sentence, random_flag=False):
    '''从结构和内容两个维度，返回最相似的句子'''

    result_1 = sentences_vector.get_vector_similarity_top_n(sentence)
    result_1 = [x + '||vector' for x in result_1]
    print 'get_vector_similarity_top_n, finished...'
    result_2 = sentences_trunk.get_structure_similarity_top_n(sentence)
    result_2 = [x + '||structure' for x in result_2]
    print 'get_structure_similarity_top_n, finished...'

    result = result_1 + result_2
    if random_flag:
        random.shuffle(result)
    # print '#'*10, sentence
    # for x in result:
    #     print x
    return result


if __name__ == '__main__':
    start_time = time.time()
    #############

    # get_similar_sentence_top_n(u'快乐是付出艰辛努力时手捧奖杯的泪水。')
    get_similar_sentence_top_n(u'亲人的爱就像阳光，总是那么温暖，那么默默无闻，细心地呵护我的成长！')
    # get_similar_sentence_top_n(u'我们在草坪上高兴地玩游戏。')

    #############
    end_time = time.time()
    print 'Finished in %.2f secondes:' % (end_time - start_time)
