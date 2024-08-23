#coding:utf8
import datetime, codecs, time
from gensim import corpora, models, similarities
from gensim.models import ldamulticore, ldamodel


def create_lda_model():
    '''作文语料LDA建模'''

    num_articles = 200000
    num_topics = 100

    # 读取语料
    fs_file = codecs.open('words_segment_%s_clean.txt' % num_articles, 'r', encoding='utf8')
    docs = fs_file.readlines()
    fs_file.close()

    # 提取文本
    texts = [doc.strip().split() for doc in docs]
    # 对文本集中所有的词汇建表
    dictionary = corpora.Dictionary(texts)
    # 把文本转换成 M a:num b:num... 的格式，用于LDA的瞄准输入
    corpus = [dictionary.doc2bow(text) for text in texts]

    starttime = datetime.datetime.now()
    print 'train model...'
    lda = ldamulticore.LdaMulticore(corpus,
        num_topics=num_topics,
        id2word=None,
        workers=3,
        chunksize=2000, 
        passes=1, 
        batch=False, 
        alpha='symmetric', 
        eta=None, 
        decay=0.5, 
        offset=1.0, 
        eval_every=None, 
        iterations=1000, 
        gamma_threshold=0.001)
    
    print 'save model... '
    lda.save('zuowen_%s_T95_iter1000' % num_articles)
    print 'log_perplexity', lda.log_perplexity(corpus)
    endtime = datetime.datetime.now()
    print 'duration:', endtime - starttime


if __name__ == '__main__':
    start_time = time.time()
    #############

    create_lda_model()

    #############
    end_time = time.time()
    print 'Finished in %.2f secondes:' % (end_time - start_time)
