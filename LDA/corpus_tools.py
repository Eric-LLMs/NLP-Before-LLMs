#coding:utf8
import codecs, time, sys

sys.path.append('../long_text_generation/')
from common import utils


def corups_clean():
    '''作文语料预处理，去除掉停用词'''

    docs = utils.get_lines_of_doc('words_segment_489200.txt')
    stop_words = utils.get_lines_of_doc('stop_words.txt')
    stop_words = set(stop_words)
    # stop_words = stop_words[1810:]

    texts = []
    for doc in docs[:200000]:
        # 停用词过滤
        words = [w for w in doc.split() if w not in stop_words]
        texts.append(' '.join(words) + '\n')

    fs_file = codecs.open('words_segment_temp.txt', 'w', encoding='utf8')
    fs_file.writelines(texts)
    fs_file.close()


if __name__ == '__main__':
    start_time = time.time()
    #############

    corups_clean()

    #############
    end_time = time.time()
    print 'Finished in %.2f secondes:' % (end_time - start_time)
