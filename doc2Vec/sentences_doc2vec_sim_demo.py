# coding:utf-8
import sys,os
import gensim
import sklearn
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from IntelligentAssistantWriting.conf.params import *
from gensim.models.doc2vec import Doc2Vec # LabeledSentence
TaggededDocument = gensim.models.doc2vec.TaggedDocument
train_data = []
def load_datasest(train_file):
    d2vmodel_docindex_data = []
    with open(train_file, 'r') as cf:
        lines = cf.readlines()
        for line in lines:
            sent = line.split('\t')[1]
            if len(sent) < 30:
                continue
            d2vmodel_docindex_data.append(sent)
        print (len(d2vmodel_docindex_data))
    global  train_data
    # y = np.concatenate(np.ones(len(docs)))
    for i, text in enumerate(d2vmodel_docindex_data):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        train_data.append(document)
    # d2vmodel_docindex_path = os.path.join(model_dir,'d2vmodel.docindex')
    # with open(d2vmodel_docindex_path,'w') as f:
    #     for data in train_data:
    #       f.write(data)
    return train_data,d2vmodel_docindex_data

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)

# def gen_d2v_corpus(self, lines):
#         with open("./data/ques2_result.txt", "wb") as fw:
#             for line in lines:
#                 fw.write(" ".join(jieba.lcut(line)) + "\n")
#
#         sents = doc2vec.TaggedLineDocument("./data/ques2_result.txt")
#         model = doc2vec.Doc2Vec(sents, size=50, window=5, alpha=0.015)
#         model.train(sents)
#
#         corpus = model.docvecs
#         np.save("./output/d2v.corpus.npy", corpus)
#
#         return np.asarray(corpus)

def train_model(train_data,model_dir, size=200, epoch_num=1):
    d2vmodel = Doc2Vec(train_data, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    d2vmodel.train(train_data, total_examples=d2vmodel.corpus_count)
    d2vmodel_path = os.path.join(model_dir,'d2vmodel.model')
    d2vmodel.save(d2vmodel_path)
    return d2vmodel


def _test():
    model_dm = Doc2Vec.load("model/model_dm_wangyi")
    test_text = ['《', '舞林', '争霸' '》', '十强' '出炉', '复活', '舞者', '澳门', '踢馆']
    inferred_vector_dm = model_dm.infer_vector(test_text)
    print (inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    return sims

def train(input_file,model_dir):
    train_data,docs = load_datasest(input_file)
    d2vmodel = train_model(train_data,model_dir, size=200, epoch_num=1)
    return d2vmodel

def load_d2vmodel(model_dir,input_file):
    train_data,d2vmodel_docindex_data = load_datasest(input_file)
    # d2vmodel_docindex_path = os.path.join(model_dir, 'd2vmodel.docindex')
    # lines = open(d2vmodel_docindex_path).readlines()
    # for line in lines:
    #     d2vmodel_docindex_data.append(line)

    d2vmodel_path = os.path.join(model_dir, 'd2vmodel.model')
    d2vmodel = Doc2Vec.load(d2vmodel_path)
    return d2vmodel,train_data, d2vmodel_docindex_data

def get_sim_sentens(d2vmodel,d2vmodel_docindex_data,query,top):
    # query_words = WordPunctTokenizer().tokenize(query.lower().replace('\n', ''))
    query_words =query.split(' ')
    inferred_vector_dm = d2vmodel.infer_vector(query_words)
    # print inferred_vector_dm
    sims = d2vmodel.docvecs.most_similar([inferred_vector_dm], topn=top)
    return sims

def _test_doc2vec_simit(test_file,model_dir,input_file):
    d2vmodel,train_data, d2vmodel_docindex_data = load_d2vmodel(model_dir,input_file)
    dic = {}
    for line in open(test_file, 'r',encoding='gbk'):
        line_s = line.split('\t')
        dic[line_s[3].replace('\n', '').replace('\r', '')] = line_s[4]
    print ('测试集有%s条人工标注的相似句对' % len(dic))
    dic_size = 0.0000
    dic_size = float(len(dic))
    dic_size_int = len(dic)
    for top in range(1,6):
        mactchcount = 0
        for k, v in dic.iteritems():
            key = u''
            value = u''
            key = '%s' % (k.replace('\r', '').replace('\n', ''))
            value = '%s' % (v.replace('\r', '').replace('\n', ''))
            sims =  get_sim_sentens(d2vmodel,train_data,str(key),top)
            iterm_res = [ ]
            for count, sim in sims:
                sentence_1 = d2vmodel_docindex_data[count]
                sentence = train_data[count]
                # print sentence_1
                # print sentence
                # print value
                    # words = ''
                    # for word in sentence[0]:
                    #     words = words + word + ' '
                    # print words, sim, len(sentence[0])
                iterm_res.append(sentence_1)
            for iterm in iterm_res:
                if value == iterm.replace('\r', '').replace('\n', ''):
                    mactchcount += 1
        print ('当推荐%s条句子给英语学习者作为参考时,测试集中人工标注总数:%s ,系统推荐的句子中有标注答案的数量:%s ,准确率为%s' % (top, dic_size_int, mactchcount, format(float(mactchcount) / float(dic_size), '.6f')))


import Levenshtein
def get_ED_sim_sentens(query,input_file,top):
     dic = {}
     with open(input_file, 'r') as cf:
         lines = cf.readlines()
         for line in lines:
             sent = line.split('\t')[1].replace('\r','').replace('\n','')
             if len(sent) < 30:
                 continue
             leng = Levenshtein.distance(query, sent)
             # print len(dic)
             if len(dic) < top:
                 dic[sent] = leng
             else :
                 for key,value in dic.items():
                    if  dic[key]  > leng:
                        dic.pop(key)
                        dic[sent] = leng
     return  dic.keys()
def _test_ED_simit(test_file,input_file):
    dic = {}
    for line in open(test_file, 'r'):
        line_s = line.split('\t')
        dic[line_s[3].replace('\n', '').replace('\r', '')] = line_s[4]
    print ('测试集有%s条人工标注的相似句对' % len(dic))
    dic_size = float(len(dic))
    dic_size_int = len(dic)
    for top in range(1,6):
        mactchcount = 0
        for k, v in dic.iteritems():
            key = u''
            value = u''
            key = '%s' % (k.replace('\r', '').replace('\n', ''))
            value = '%s' % (v.replace('\r', '').replace('\n', ''))
            sim_sentences =  get_ED_sim_sentens(str(key),input_file,top)
            for sim_sentence in sim_sentences:
                if value == sim_sentence.replace('\r', '').replace('\n', ''):
                    mactchcount += 1
        print ('当推荐%s条句子给英语学习者作为参考时,测试集中人工标注总数:%s ,系统推荐的句子中有标注答案的数量:%s ,准确率为%s' % (top, dic_size_int, mactchcount, format(float(mactchcount) / float(dic_size), '.6f')))

if __name__ == '__main__':
    conf = create_params()
    model_dir = os.path.join(conf.dir_root, 'temp/train_temp/train_sentences_d2w')
    train_file = os.path.join(conf.dir_root,  'myworkspace/data/MSRParaphraseCorpus/msr_paraphrase_data_sentences_without_simtext')
    test_file =  os.path.join(conf.dir_root, 'myworkspace/data/test_similar')
    # load_datasest(train_file, model_dir)
    # train(train_file, model_dir)
    _test_doc2vec_simit(test_file, model_dir,train_file)
    # test_ED_simit(test_file, train_file)
    # print ('ok')
    # sims = test()
    # for count, sim in sims:
    #     sentence = x_train[count]
    #     words = ''
    #     for word in sentence[0]:
    #         words = words + word + ' '
    #     print words, sim, len(sentence[0])
    # import  Levenshtein
    texta = 'book good math'
    textb = 'text book gook'
    print (Levenshtein.distance(texta, textb))

    print ('ok')