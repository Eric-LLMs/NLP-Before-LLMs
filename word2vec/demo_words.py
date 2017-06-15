# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import  os,sys
import re
from gensim.models import word2vec
from nltk.corpus import wordnet as wn
from nltk.parse import stanford
from IntelligentAssistantWriting.conf.params import *
import nltk

from nltk.stem import WordNetLemmatizer


config = create_params_lexile()
model_dir = os.path.join(config.dir_root,'temp/train_temp/train_words')
input_file = os.path.join(config.dir_root,'myworkspace/data/train_data_word2vec')
w2vmodel_path = os.path.join(config.dir_root,'temp/train_temp/train_sentences/w2v_model.model')

def train_word2vec(train_file,w2vmodel_path):
    sentences = word2vec.Text8Corpus(train_file)
    model = word2vec.Word2Vec(sentences,min_count=1,window=5,size=200)
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

# stan_jar_dir = '/home/temp/stanfordParserFull'
stan_jar_dir = os.path.join(config.dir_root,'temp/stanfordParserFull')
parser_jar_path = os.path.join(stan_jar_dir,'stanford-parser.jar')
models_jar_path = os.path.join(stan_jar_dir,'stanford-parser-3.8.0-models.jar')
model_file = os.path.join(stan_jar_dir,'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
eng_parser = stanford.StanfordDependencyParser(path_to_jar=parser_jar_path,path_to_models_jar=models_jar_path,model_path=model_file)
parser = stanford.StanfordParser(path_to_jar=parser_jar_path,path_to_models_jar=models_jar_path,model_path=model_file)
stop_words_file = os.path.join(config.dir_root,'myworkspace/data/stop_words_for_expkeywords')
s_words = [] #停用词表
#tree
def parser_sent(query):
    sentences = parser.raw_parse_sents(query)
    print (sentences)
    for line in sentences:
        for sentence in line:
            print (sentence)

def get_exp_words(query):
    words_res = []
    # 先找'nsubj' 主谓关系 如果含有NN 则加入字典列表
    # amod/JJ/JJR/JJS-->adj  RB/RBR/RBS-->adv  auxpass/VB/VV --> v
    # w_tags =['NN','amod','JJ','JJR','JJS','RB','RBR','RBS','auxpass','VB','VV','VBP','VBP']
    w_tags = [ 'NN','amod', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'auxpass', 'VB', 'VV', 'VBP', 'VBP']
    result = list(eng_parser.parse(query.replace('\'','').replace(',',' ').replace('.',' ').replace('!',' ').replace(';',' ').replace('?',' ').split( )))
    words_list = []
    for row in result[0].triples():
        # print(row);
        # print(row[0][0],row[0][1],'|',row[1],'|',row[2][0],row[2][1])
        try:
            w_tag = row[2][1].replace('\'', '')
            if w_tag in w_tags:
                word = row[2][0].replace('\'', '')
                if word not in words_list:
                  words_list.append(word)
            w_tag = row[0][1].replace('\'', '')
            if w_tag in w_tags:
                word = row[0][0].replace('\'', '')
                if word not in words_list:
                  words_list.append(word)
        except Exception as e:
            print (e)

    global s_words
    s_words = []
    for line in open(stop_words_file, "rb"):
        line =str(line,'UTF-8')
        s_words.append(line.replace('\r', '').replace('\n', ''))
    for w in words_list:
        if w in s_words:
            continue
        words_res.append(w)

    return  words_res

dict_globle = {'peking univercity':1}
def load_dic(dic_file = os.path.join(config.dir_root,'myworkspace/data/ALL_List')):
    for line in open(dic_file,'rb'):
        line = str(line, encoding = "utf-8")
        line = line.replace('\n', '').replace('\r','')
        dict_globle[line] = 1;

lemmatizer = WordNetLemmatizer()
def words_expansion(query):
    global lemmatizer
    global dict_globle
    dic_file = os.path.join(config.dir_root,'myworkspace/data/ALL_List')
    load_dic(dic_file)
    w_list = get_exp_words(query)
    result_temp = []
    result = []
    for w in w_list:
        e_list = []
        e_list.append(w)
        try:
           syn_list = get_synonym_words(w)
           for item in syn_list:
               if len(item) > 0:
                  e_list.append(item)
        except Exception as e:
           print (e)
        try:
            e_list.append(get_most_similar_words(w2vmodel_path,w,10))
        except Exception as e:
            print  (e)
        # print e_list
        result_temp.append(e_list)

    for words in result_temp:
        wordArr = []
        wordArr.append(words[0])
        for i in range(1, len(words)):
                for word in words[i]:
                    if word in s_words: #在停用词表中过滤掉
                        continue
                    word_o = lemmatizer.lemmatize(word).lower()
                    if (word.lower() in dict_globle.keys()) or (word_o in dict_globle.keys()):
                        wordArr.append(word.lower())
        result.append(wordArr)
    return w_list,result

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
    print ('1')

import networkx as nx
import  random
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
    # demo_show()
    #训练word2vev 模型
    # train_word2vec(input_file,w2vmodel_path)
    #word2vec 词中取得关联词
    # print get_most_similar_words(w2vmodel_path,'sad',10) #关联词
    #wordnet 取得近义词、同义词、上下位词
    # get_synonym_words('sky') # norn 同一类别、上下位词
    # print get_synonym_words('beautiful')  # ad 同义词、近义词
    # get_synonym_words('bite')#类别、上下位词
    # sets=("The city is a happy blending of town and country.", "What is your name?", "Not only is he happy to meet up, he wants to show us round.")
    # parser_sent(sets)
    # query = 'I am so happy'
    # print  get_exp_words(query)
    # query = 'The city is a happy blending of town and country,I eat an apple.'
    # query = 'A night of peaceful slumber makes me feel so happy .'
    # query = 'The beautiful sky make me so happy.'

    query = ' A healthy diet is very important for people.'
    # query = 'So I am very happy to be Chinses .'
    w_list, result = words_expansion(query)
    print (w_list)
    print  (result)
    drowWord2VecGraph()
    print ('ok')
