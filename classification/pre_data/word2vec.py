# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import re
from gensim.models import word2vec
from nltk.corpus import wordnet as wn
from IntelligentAssistantWriting.conf.params import *

#############################   windows path #####################################
config = create_params_calssification()
# model_dir = os.path.join(config.dir_root,'classification/train_words')
input_file = os.path.join(config.dir_root,'train_data_word2vec')
w2vmodel_path = os.path.join(config.dir_root,'w2v_model.model')
#######################################################################

def train_word2vec(train_file,w2vmodel_path):
    sentences = word2vec.Text8Corpus(train_file)
    model = word2vec.Word2Vec(sentences,min_count=1,window=5,size=128)
    y = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)
    print('most_similar')
    for item in y:
        print (item[0], item[1])
    y = model.doesnt_match("breakfast cereal dinner lunch".split())
    print ('doesnt_match')
    print (y)
    y = model.similarity('woman', 'man')
    print('similarity')
    print (y)

    print ("--------\n")
    model.save(w2vmodel_path)

# error if not in
def get_most_similar_words(model_path,word,num):
    wv2model = word2vec.Word2Vec.load(w2vmodel_path)
    list = wv2model.most_similar(word, topn=num)
    result = []
    # return  list
    for item in list:
         print (item)
         result.append(item[0])
    return  result

# error if not in
def get_synonym_words(word):
    #tong yi ci ji
    sub_sets = []
    for item in wn.synsets(word):
        mark = re.findall("Synset\(\'(.*)\'\)", str(item))[0]
        sub_set = wn.synset(mark).lemma_names()
        for item in sub_set:
            if word == item:
                continue
            sub_sets.append(item)
    print (sub_sets)
    #xiaing si ji
    similar = wn.synsets(word)[0]
    similar_sets = []
    for item in  similar.similar_tos():
        similar_set = re.findall("Synset\(\'(.*?)\.", str(item))[0]
        similar_sets.append(similar_set)
    print (similar_sets)
    return  sub_sets,similar_sets

def demo_show():
    # dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First', 'book': ''};
    # print dict['Name']
    # if dict.has_key('NoWords') == 0:
    #     print 'do not has NoWords'
    # if dict.has_key('Name') == 1:
    #     print 'has Name'
    # print(lemmatizer.lemmatize("urban_center"))
    # print(lemmatizer.lemmatize("cats"))
    # print(lemmatizer.lemmatize("dead"))
    # print(lemmatizer.lemmatize("closed"))
    wv2model = word2vec.Word2Vec.load(w2vmodel_path)
    print (wv2model['Agamemnon'])
    # word2vec 词中取得关联词
    # print (get_most_similar_words(w2vmodel_path,'sad',10)) #关联词
    # wordnet 取得近义词、同义词、上下位词
    # get_synonym_words('sky') # norn 同一类别、上下位词
    # print (get_synonym_words('beautiful'))  # ad 同义词、近义词
    # get_synonym_words('bite')#类别、上下位词
    # drowWord2VecGraph()

    wv2model = word2vec.Word2Vec.load(w2vmodel_path)
    for i in wv2model.wv.vocab:
        # print(wv2model[jjj])
        jjj = wv2model.wv.vocab[i].index
        print(i, jjj, wv2model[i])
    print ('1')

import networkx as nx
import matplotlib.pyplot as plt
def drowWord2VecGraph():
    list_graphshow = {"sad": 0.5694999694824219,
                      "wonderful": 0.5681723356246948,
                      "nice": 0.530474066734314,
                      "comfortable": 0.5075299739837646,
                      "fun": 0.49314290285110474,
                      "pleasant": 0.49166160821914673,
                      "exciting": 0.4887125492095947,
                      "tired": 0.4827210009098053,
                      "happily": 0.47587382793426514,
                      "proud": 0.4724070131778717, }
    list_graphshow_1 = {'lonely': 0.6279479265213013,
                        'unhappy': 0.5777831077575684,
                        'depressed': 0.5655499696731567,
                        'relaxed': 0.5576751232147217,
                        'clever': 0.5292290449142456,
                        'confused': 0.5171307921409607,
                        'humorous': 0.5168302059173584,
                        'angry': 0.5157681703567505,
                        'nice': 0.5117100477218628}
    G = nx.Graph()
    # for  v in nx.barabasi_albert_graph(10, 2, seed=1).edges():
    for v in list_graphshow.keys():
        print  (v,list_graphshow[v])
        G.add_edge("happy", v, weight=list_graphshow[v]*100-43)
    # G.add_edge("sad", 'depressed', weight=list_graphshow_1['depressed'] * 100 - 45)
    # G.add_edge("sad", 'angry', weight=list_graphshow_1['angry'] * 100 - 45)
    # G.add_edge("sad", 'lonely', weight=list_graphshow_1['lonely'] * 100 - 45)
    # G.add_edge("sad", 'confused', weight=list_graphshow_1['confused'] * 100 - 45)
    # for v in list_graphshow.keys():
    #     print  v, list_graphshow[v]
    #     G.add_edge("sad", v, weight=list_graphshow[v] * 100 - 45)
    pos = nx.spring_layout(G, iterations=20)
    edgewidth = []
    # for (u, v, d) in G.edges(data=True):
    #     edgewidth.append(round(G.get_edge_data(u, v).values()[0] * 20, 2))
    # nx.draw_networkx_edges(G, pos, width=edgewidth)
    # nx.draw_networkx_nodes(G, pos)
    nx.draw(G, node_color='w', with_labels=True,node_size = 1000)
    plt.show()

if __name__=='__main__' :
    # 训练word2vev 模型
    # train_word2vec(input_file,w2vmodel_path)
    wv2model = word2vec.Word2Vec.load(w2vmodel_path)
    print(wv2model['beautiful'])
    pass
