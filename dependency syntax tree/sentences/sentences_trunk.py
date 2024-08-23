#coding=utf8
from pymongo import MongoClient
import copy, time, sys
import sentences_levenshtein

# sys.path.append('../')
from src.common import utils


def get_sentence_trunk(m_dict):
    '''提取句子主干结构，参数：句子库中的单个记录 | 句法树层次深度'''

    DEPTH = 3

    parsing_list = m_dict['parsing']
    for x in parsing_list:      # 求树的根节点
        if x['parent'] == -1:
            ROOT_ID = x['id']
            break

    # print m_dict['content']
    # print parsing_list

    sentence_tree = [[ROOT_ID]]     # 句法分析树，各层的节点
    while True:
        temp_list = []
        for ID in sentence_tree[-1]:
            for x in parsing_list:
                if x['parent'] == ID:
                    temp_list.append(x['id'])
        if temp_list:
            sentence_tree.append(temp_list)
        else:
            break

    temp_list, result = [], []      # 句法分析树，从上至下，树形结构不断扩充
    for x in sentence_tree:
        temp_list += x
        temp_list.sort()
        result.append(copy.copy(temp_list))

    # 打印句法树
    # for x in xrange(len(result)):
    #     print len(result[x]),':',
    #     for y in result[x]:
    #         print y,
    #     print

    # 考虑到树的层级不超过depth的异常情况
    try:
        result = [parsing_list[ID]['relate'] for ID in result[DEPTH-1]]
    except Exception, e:
        result = [parsing_list[ID]['relate'] for ID in result[-1]]
    
    return result


def update_sentences_length():
    '''给句子库添加句子长度字段'''

    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['sentences']

    num = 0
    items = coll.find({'trunk_depth_3': {'$exists': True}})    # 找出所有具有句子结构主干的句子
    for item in items[:]:
        content = item['content']
        # print len(content)
        coll.update({'content': content}, {'$set': {'length': len(content)}})

        num += 1
        if num % 10000 == 0:
            print num

    mc.close()


def update_sentences_trunk():
    '''给句子库添加句子结构主干字段'''

    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['sentences']

    num = 0
    items = coll.find()
    for item in items[:1]:
        if not item.has_key('trunk_depth_3'):
            sentence_trunk = get_sentence_trunk(item)
            # print sentence_trunk
            coll.update({'content': item['content']}, {'$set': {'trunk_depth_3': sentence_trunk}})

        num += 1
        if num % 10000 == 0:
            print num

    mc.close()


def update_sentences_structure_words():
    '''给句子添加结构主干词语字段'''

    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['sentences']

    result = []
    num = 0
    items = coll.find({'trunk_depth_3': {'$exists': True}})    # 找出所有具有句子结构主干的句子
    for item in items[:1]:
        trunk_A = item['trunk_depth_3']
        key_words_A = get_structure_words(trunk_A, item)
        coll.update({'content': item['content']}, {'$set': {'structure_words': key_words_A}})

        # print item['content']
        # for k, v in key_words_A:
        #     print k, v, type(k)

        num += 1
        if num % 10000 == 0:
            print num

    mc.close()


def clean_sentences_trunk():
    '''清理句子库，删除结构树只有一层的句子'''

    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['sentences']

    num = 0
    items = coll.find({'trunk_depth_3': {'$exists': True}})
    for item in items[:]:
        sentence_trunk = item['trunk_depth_3']
        if len(sentence_trunk) == 1:
            # print sentence_trunk
            coll.remove({'content': item['content']})

        num += 1
        if num % 10000 == 0:
            print num

    mc.close()


def get_structure_words(my_sentence_trunk, my_sentence_parsing):
    '''找出句子中“主谓宾”词语'''

    my_sentence_words = [x['cont'] for x in my_sentence_parsing['parsing']]
    my_sentence_relate = [x['relate'] for x in my_sentence_parsing['parsing']]

    my_key_words = []
    try:
        index = my_sentence_trunk.index('SBV')
        my_key_words.append(('SBV', my_sentence_words[my_sentence_relate.index('SBV')]))
    except Exception, e:
        pass

    try:
        index = my_sentence_trunk.index('HED')
        my_key_words.append(('HED', my_sentence_words[my_sentence_relate.index('HED')]))
    except Exception, e:
        pass

    try:
        index = my_sentence_trunk.index('VOB')
        my_key_words.append(('VOB', my_sentence_words[my_sentence_relate.index('VOB')]))
    except Exception, e:
        pass

    return my_key_words


def get_structure_similarity_top_n(sentence):
    '''返回与句子库中结构相似度最高的若干句子'''

    # 句法分析，求句子结构
    my_sentence_parsing = utils.sentence_parsing(sentence)
    my_sentence_trunk = get_sentence_trunk(my_sentence_parsing)
    my_key_words = get_structure_words(my_sentence_trunk, my_sentence_parsing)

    my_key_words = [x for x in my_key_words if x[0] in ['HED', 'SBV']]
    for k,v in my_key_words:
        print  k, v

    mc = MongoClient('127.0.0.1')
    coll = mc['zuowen']['sentences_new']

    result = []
    # 找出所有具有句子结构主干的句子，且对句子长度进行了限制
    items = coll.find({'trunk_depth_3': {'$exists': True},
                'length': {'$gt': len(sentence) * 0.5, '$lt': len(sentence) * 2}})
    for item in items[:]:
        key_words_A = item['structure_words']
        key_words_A = [x for x in key_words_A if x[0] in ['HED', 'SBV']]

        # 句子主干词语保持一致
        if ''.join([x[1] for x in my_key_words]) == ''.join([x[1] for x in key_words_A]):
            trunk_A = item['trunk_depth_3']
            similarity = sentences_levenshtein.levenshtein_ratio(my_sentence_trunk, trunk_A)
            if similarity > 0.5:
                # print similarity, item['content']
                result.append((similarity, item['content']))

            # distance = sentences_levenshtein.levenshtein_distance(my_sentence_trunk, trunk_A)
            # if distance < 5:
            #     print distance, item['content'], trunk_A
            #     result.append((distance, item['content']))

    mc.close()

    result.sort(reverse=True)
    if len(result) >= 5:
        result = [x[1] for x in result[:5]]
    else:
        result = [x[1] for x in result]

    # for x in result:
    #     print x

    return result


if __name__ == '__main__':
    start_time = time.time()
    #############

    # update_sentences_trunk()
    # update_sentences_structure_words()
    # update_sentences_length()
    # clean_sentences_trunk()
    # get_structure_similarity_top_n(u'大山和大海都是我们的好朋友。')

    #############
    end_time = time.time()
    print 'Finished in %.2f secondes:' % (end_time - start_time)
