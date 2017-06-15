#coding:utf8
from gensim import corpora, models, similarities
from gensim.models import ldamulticore, ldamodel
import codecs, time, sys
from pymongo import MongoClient

# sys.path.append('../')
from src.common import utils


def print_sample(texts, dictionary, corpus, index):
    '''打印样例数据'''

    print len(texts), len(dictionary), len(corpus)
    # for x in texts[index]:
        # print x
    print 'dictionary', dictionary[index]
    print 'corpus', corpus[index]


def get_topic_words(topicid_list, lda, dictionary, num_topic_words):
    '''返回 topic_id 序列对应的主题词序列，其中每个 topic_id 各对应一个主题词序列'''

    # 获取某个 topic_id 对应的主题词 (index, probability) 序列
    word_index_list = []
    word_probalibity_list = []
    for topic_id in topicid_list:
        # Return a list of (word, probability) 2-tuples for the most probable words in topic topicid.
        twords = lda.show_topic(topic_id, num_topic_words)
        index = map(lambda i : int(i[0]), twords)
        probability = map(lambda i : float(i[1]), twords)
        word_index_list.append(index)
        word_probalibity_list.append(probability)

    # 获取某个 topic_id 对应的主题词 index 序列所对应的主题词序列
    topic_words = []
    for index in word_index_list:
        twords = map(lambda i : dictionary[i], index)
        topic_words.append(twords)

    return topic_words, word_probalibity_list


def print_topics_of_document(lda, dictionary, num_topic_words, topic_distribution):
    '''打印某个文档对应的主题分布情况'''

    # print topic_distribution
    
    topicid_list = [topic_id for topic_id, topic_probability in topic_distribution]
    topic_words = get_topic_words(topicid_list, lda, dictionary, num_topic_words)    

    for i in xrange(len(topic_distribution)):
        print '#%d: ' % topic_distribution[i][0],
        for m in xrange(num_topic_words):
            # print topic_words[i][m], ':', str(word_probalibity_list[i][m]), '/',
            print topic_words[i][m],
        print


def get_topics_of_new_document(topic_distribution):
    '''打印新文档返回的主题分布情况'''

    # print topic_distribution

    fs_file = codecs.open('static/topic_model_100.txt', 'r', encoding='utf8')
    topics = [x.strip() for x in fs_file.readlines()]
    fs_file.close()

    topicid_list = [topic_id for topic_id, topic_probability in topic_distribution]

    result = []
    for topic_id in topicid_list:
        # print '#%s' % topic_id, LDA[topic_id]
        result.append(topics[topic_id])

    if len(result) >= 2:
        result = result[:2]

    # for x in result:
    #     print x

    return result


def corpus_preparation(article):
    '''对某篇文章进行预处理(分词，去除停用词)，以计算其主题分布情况'''

    # 1. 分词
    word_segment = utils.word_segmentation(article)
    doc = ' '.join([' '.join(x) for x in word_segment])
    doc = doc.decode('utf8')

    # 2. 停用词过滤
    stop_words = utils.get_lines_of_doc('static/stopwords_topics_module.txt')
    stop_words = set(stop_words)
    words = [w for w in doc.split() if w not in stop_words]
    # print doc
    # print ' '.join(words)
    return ' '.join(words)


def get_topic_by_article(article):
    '''根据LDA模型，计算某篇作文的主题'''

    num_articles = 200000    # 文档数量
    num_topics = 100         # 主题数量
    num_topic_words = 20    # 某主题下的前20个主题词

    # 1. 获取lda模型中的词汇表
    fs_file = codecs.open('static/words_segment_%s_clean.txt' % num_articles, 'r', encoding='utf8')
    docs = fs_file.readlines()
    fs_file.close()

    # 提取文本
    texts = [doc.strip().split() for doc in docs]
    # 对文本集中所有的词汇建表
    dictionary = corpora.Dictionary(texts)

    # 2. 新文章，预处理
    doc_prepared = corpus_preparation(article)
    new_texts = [doc_prepared.strip().split()]

    # 3. 把文本转换成 M a:num b:num... 的格式，用于LDA的瞄准输入
    corpus = [dictionary.doc2bow(text) for text in new_texts]
    # print_sample(texts, dictionary, corpus, len(corpus)-1)

    # 4. 获取主题分布
    lda = ldamulticore.LdaMulticore.load('static/zuowen_%s_T95_iter1000' % num_articles)
    topic_distribution = lda.get_document_topics(corpus[0]) # a list of (topic_id, topic_probability) 2-tuples

    # 5. 对主题分布概率排序
    # print topic_distribution
    topic_distribution = [(x[1], x[0]) for x in topic_distribution]
    topic_distribution.sort(reverse=True)
    topic_distribution = [(x[1], x[0]) for x in topic_distribution]

    # 6. 输出该文档的主题分布情况
    result = get_topics_of_new_document(topic_distribution)
    # print_topics_of_document(lda, dictionary, num_topic_words, topic_distribution)

    return result


def get_all_topics():
    '''根据LDA模型，获取所有训练后的主题'''

    num_articles = 200000    # 文档数量
    num_topics = 100         # 主题数量
    num_topic_words = 20    # 某主题下的前20个主题词

    fs_file = codecs.open('words_segment_%s_clean.txt' % num_articles, 'r', encoding='utf8')
    docs = fs_file.readlines()
    fs_file.close()

    texts = [doc.strip().split() for doc in docs]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print_sample(texts, dictionary, corpus, 0)

    lda = ldamulticore.LdaMulticore.load('zuowen_%s_T95_iter1000' % num_articles)

    # 输出所有主题分布情况
    # mc = MongoClient('127.0.0.1')
    # coll = mc['zuowen']['LDA']

    topic_words, word_probalibity_list = get_topic_words(xrange(num_topics), lda, dictionary, num_topic_words)
    for i in xrange(num_topics):
        print '#%d: ' % i,
        for m in xrange(num_topic_words):
            print topic_words[i][m], ':', str(word_probalibity_list[i][m]), '|',
            print topic_words[i][m],
        print

        # 保存主题到数据库
    #     word_probalibity = [(topic_words[i][m], word_probalibity_list[i][m]) for m in xrange(num_topic_words)]
    #     dic = {}
    #     dic['topic_id'] = i
    #     dic['topic_words'] = word_probalibity
    #     coll.save(dic)

    # mc.close()


if __name__ == '__main__':
    start_time = time.time()
    #############

    # get_all_topics()
    get_topic_by_article(u'今天是周末，天气是很好的。我们的心情也是很高兴的。天空是蓝色的，非常漂亮。春天是很适合出去游玩的季节。于是，我邀请我的好朋友大卫一起去春游。他还带了他的一个宠物小狗。这个小狗很活泼可爱。我们都很喜欢它。我们高高兴兴地背着包来到目的地。那里有很多绿色和好看的草坪。很多鲜艳的花朵也都开放了。这里真是一个风景迷人的地方。我们把包里的食物和玩的东西放到草坪上。在开始的时候，我们在草坪上高兴地玩游戏。还有吃了很多美味的东西。两个人还分享了。我给他一颗红的苹果。他送了我一瓶冰凉的汽水。心情很开心的。后来我和他开始踢足球了。我们踢得很热烈。最后，因为我们都很疲累了，所以就回去了。这次春游真是给我留下很难忘的印象。')

    #############
    end_time = time.time()
    print 'Finished in %.2f secondes:' % (end_time - start_time)
