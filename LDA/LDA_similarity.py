#coding=utf8
import numpy
import codecs, time
from pymongo import MongoClient


def get_common_words_num(a_list, b_list):
    '''求两个列表的重复元素个数'''

    intersection = set(a_list).intersection(set(b_list))
    return len(intersection)


def cos_calculate(x, y):
    '''计算两个向量的余弦夹角'''

    x = numpy.array(x)
    y = numpy.array(y)
    len_x = numpy.sqrt(x.dot(x))
    len_y = numpy.sqrt(y.dot(y))
    return x.dot(y) / (len_x * len_y)


def get_my_topic_vector(topic_words, extra_words=[]):
    '''计算主题向量'''

    result = [0.0 for x in xrange(200)]

    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['words']
    N = 0
    for w in topic_words:
        item = coll.find_one({'content': w, 'vector': {'$exists': True}})
        if item:
            N += 1
            vector = item['vector']
            for i in xrange(200):
                result[i] += vector[i]

    M = 0
    for w in extra_words:
        item = coll.find_one({'content': w, 'vector': {'$exists': True}})
        if item:
            M += 1
            vector = item['vector']
            for i in xrange(200):
                result[i] += vector[i] * 2.0

    mc.close()

    if N > 0 or M > 0:
        result = [x / (N + M * 2) for x in result]
    
    return result


def get_topic_similarity_top_n(original_topics, extra_words=[]):
    '''返回主题分布中相似度最高的若干主题'''

    # 求当前主题向量
    original_vectors = []
    for my_topic in original_topics:
        my_topic_words = my_topic.split()
        original_vectors.append(get_my_topic_vector(my_topic_words, extra_words))

    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['LDA']

    result = []
    items = coll.find()    # 找出所有主题
    for item in items:
        m_topic_words = [x[0] for x in item['topic_words']]

        if ' '.join(m_topic_words) not in original_topics:
            # 主题库中某个主题的向量
            vector_A = get_my_topic_vector(m_topic_words)

            sim = 0.0
            for i in xrange(len(original_vectors)):  # original_vectors 长度最大为5
                similarity = cos_calculate(vector_A, original_vectors[i])
                sim += similarity * (1 - 0.2 * i)
                
            result.append((sim, m_topic_words))

    mc.close()
    result.sort(reverse=True)

    final_result = []
    for m_topic in result[:5]:
        topic_words_A = m_topic[1]

        count = 0
        count += get_common_words_num(topic_words_A, extra_words) * 2
        for i in xrange(len(original_topics)):
            my_topic_words = original_topics[i].split()
            common_words_num_A = get_common_words_num(topic_words_A, my_topic_words)
            count += common_words_num_A * (1 - 0.2 * i)

        final_result.append((count, topic_words_A))

    final_result.sort(reverse=True)
    final_result = [' '.join(x[1]) for x in final_result[:2]]

    # for x in final_result:
    #     print x
    
    return final_result


if __name__ == '__main__':
    start_time = time.time()
    #############

    extra_words = [u'老师', u'同学']
    original_topics = ['书 读书 故事 知识 书籍 阅读 爱 喜欢 小说 童话 快乐 文字 生活 智慧 世界 作文 内容 道理 海洋 主人公',
                        '老师 同学 作业 学习 成绩 上课 考试 教室 数学 年级 学校 语文 题 第一 学生 天 英语 试卷 作文 下课']
    get_topic_similarity_top_n(original_topics, extra_words)

    #############
    end_time = time.time()
    print 'Finished in %.2f secondes:' % (end_time - start_time)
