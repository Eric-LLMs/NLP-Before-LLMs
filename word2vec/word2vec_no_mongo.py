#coding=utf8
import numpy
import threading
import time, codecs, sys

# sys.path.append('../')
from src.common import utils


def clean_vectors_bin():
    '''删除掉 vectors.bin 文件中的一些异常词，如乱码字、繁体词，空格等'''

    # 如果打开失败，说明存在非法符号，需要用默认编码打开
    filename = 'vectors.bin'
    fs_infile = codecs.open(filename, 'r', encoding='utf8') 
    # fs_infile = codecs.open(filename, 'r') 
    lines = fs_infile.readlines()
    fs_infile.close()

    # 1 定位出现异常的位置
    # for x in xrange(len(lines[:])):
        # line = lines[x]

        # 1.1 针对乱码符号
        # try:
        #     line = line.decode('utf8')
        # except Exception, e:
        #     print x
        #     print lines[x][:120]
        #     continue
        
        # 1.2 针对空格等情况
        # ll = line.split()
        # if len(ll) != 201:
        #     print 'line NO.:', x, '==>', line[:120]

    # 2 删除异常数据
    # out_set = [167205]
    # print len(lines)
    # result = []
    # for x in xrange(len(lines)):
    #     if x not in out_set:
    #         result.append(lines[x].decode('utf8'))

    # print len(result)
    # fs_outfile = codecs.open('output.txt', 'w', encoding='utf8')
    # fs_outfile.writelines(result)
    # fs_outfile.close()


return_dict = {}

class CosCalculator(threading.Thread):
    '''新建独立线程，计算每个词的相近词'''

    global return_dict

    def __init__(self, word, vectors):
        threading.Thread.__init__(self)
        self.word = word
        self.vectors = vectors


    def cos_vector(self, x, y):
        x = numpy.array(x)
        y = numpy.array(y)

        len_x = numpy.sqrt(x.dot(x))
        len_y = numpy.sqrt(y.dot(y))
        return x.dot(y) / (len_x * len_y)


    def run(self):
        words = [x[0] for x in self.vectors]    # 获取 word2vec 中的词列表

        # 查看词表
        # for w in words:
        #     print w.encode('utf8')
        # return_dict[self.word] = []
        # return None

        if self.word not in words:
            return_dict[self.word] = []
            return None

        i = words.index(self.word)  # 计算该词与其它词的余弦值
        result = []
        for j in xrange(len(self.vectors)):
            result.append((self.cos_vector(self.vectors[i][1:], self.vectors[j][1:]), self.vectors[j][0]))
        result.sort(reverse=True)

        temp_result = [v for k, v in result[1:21]]
        return_dict[self.word] = temp_result


def get_similar_words(word_list):
    '''对word_list中的每一个词新建一个线程，获取其相近词'''

    global return_dict
    return_dict = {}

    # 从 vectors.bin 读入词向量模型，得到词向量矩阵
    in_file = codecs.open('static/vectors.bin', 'r', encoding='utf8')
    lines = in_file.readlines()
    in_file.close()

    vectors = []
    for line in lines[1:]:
        temp_list = line.split()
        vec = [temp_list[0]]
        vec += [float(x) for x in temp_list[1:]]
        vectors.append(vec)

    # 开启多线程，获取每个词的相近词
    for word in word_list:
        calculator = CosCalculator(word, vectors)
        calculator.start()

    while len(return_dict) != len(word_list):
        time.sleep(0.1)

    return return_dict


if __name__ == '__main__':
    start_time = time.time()
    #############

    # clean_vectors_bin()
    # words = [u'开心', u'痛苦']
    words = [u'开心']
    # words = [u'来到', u'北京', u'清华大学', u'做', u'主持']
    result = get_similar_words(words)
    for k, v in result.items():
        print '#'*10, k
        utils.print_list(v)

    #############
    end_time = time.time()
    print 'Finished in %.2f secondes:' % (end_time - start_time) 
