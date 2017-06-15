# -*- coding: utf-8 -*-
import os, sys
import json
from gensim import corpora,models,similarities,utils
from IntelligentAssistantWriting.demo.utils import  *
from IntelligentAssistantWriting.demo.topic_demo import filter_char
from IntelligentAssistantWriting.demo.sentences_summary import extract_top_sentences


############################# windows huan jing ###########################
train_model_dir = 'G:/home/temp/train_temp/train_article_lda'
data_path = 'G:/home/myworkspace/data/train_data'
#########################################################################

# 载入字典
# dictionary = corpora.Dictionary.load(os.path.join(train_model_dir, "all.dic"))
# # 载入TFIDF模型和索引
# tfidfModel = models.TfidfModel.load(os.path.join(train_model_dir, "allTFIDF.mdl"))
# indexTfidf = similarities.MatrixSimilarity.load(os.path.join(train_model_dir, "allTFIDF.idx"))
# # 载入LDA模型和索引
# ldaModel = models.LdaModel.load(os.path.join(train_model_dir, "allLDA50Topic.mdl"))
# indexLDA = similarities.MatrixSimilarity.load(os.path.join(train_model_dir, "allLDA50Topic.idx"))

def filter_char(content):
    if ('.txt' in content) or (')' in content) or ('(' in content) or ('<' in content)or ('>' in content):
        content = ''
    else:
      content = content.replace(',','').replace('.','').replace(';','').replace('<','').replace('>','')
    return content

def get_simst_topic(train_model_dir,query):
    dictionary = corpora.Dictionary.load(os.path.join(train_model_dir, "all.dic"))
    tfidfModel = models.TfidfModel.load(os.path.join(train_model_dir, "allTFIDF.mdl"))
    indexTfidf = similarities.MatrixSimilarity.load(os.path.join(train_model_dir, "allTFIDF.idx"))
    ldaModel = models.LdaModel.load(os.path.join(train_model_dir, "allLDA400Topic.mdl"))
    indexLDA = similarities.MatrixSimilarity.load(os.path.join(train_model_dir, "allLDA400Topic.idx"))
    # query就是测试数据，先切词
    # query = "this a happy day,I am so happy"
    query = query
    content = (query.lower()).split(' ')
    # print content
    query_bow = dictionary.doc2bow(filter(lambda x: len(x) > 0, map(filter_char, content)))
    # print query_bow
    # 使用TFIDF模型向量化
    tfidfvect = tfidfModel[query_bow]

    # print tfidfvect
    # 然后LDA向量化，因为我们训练时的LDA是在TFIDF基础上做的，所以用itidfvect再向量化一次
    ldavec = ldaModel[tfidfvect]

    # print ldavec
    # TFIDF相似性
    simstfidf = indexTfidf[tfidfvect]
    # LDA相似性
    simlda = indexLDA[ldavec]

    # print simstfidf
    # print simlda

    # print max(simlda.items(),key=lambda x:x[1])
    # print  simlda.max()
    # print 'ok'
    return simstfidf, simlda

def get_simst_content(train_model_dir,query):
    # 载入字典
    dictionary = corpora.Dictionary.load(os.path.join(train_model_dir, "all.dic"))
    # 载入TFIDF模型和索引
    tfidfModel = models.TfidfModel.load(os.path.join(train_model_dir, "allTFIDF.mdl"))
    indexTfidf = similarities.MatrixSimilarity.load(os.path.join(train_model_dir, "allTFIDF.idx"))
    # 载入LDA模型和索引
    ldaModel = models.LdaModel.load(os.path.join(train_model_dir, "allLDA400Topic.mdl"))
    indexLDA = similarities.MatrixSimilarity.load(os.path.join(train_model_dir, "allLDA400Topic.idx"))

    # query就是测试数据，先切词
    # query = "this a happy day,I am so happy"
    query = query
    content = (query.lower()).split(' ')
    # print content
    query_bow = dictionary.doc2bow(filter(lambda x: len(x) > 0, map(filter_char, content)))
    # print query_bow
    # 使用TFIDF模型向量化
    tfidfvect = tfidfModel[query_bow]

    # print tfidfvect
    # 然后LDA向量化，因为我们训练时的LDA是在TFIDF基础上做的，所以用itidfvect再向量化一次
    ldavec = ldaModel[tfidfvect]

    # print ldavec
    # TFIDF相似性
    simstfidf = indexTfidf[tfidfvect]
    # LDA相似性
    simlda = indexLDA[ldavec]

    # print simstfidf
    # print simlda

    # print max(simlda.items(),key=lambda x:x[1])
    # print  simlda.max()
    # print 'ok'
    return simstfidf, simlda

def get_query_simst(query,dbconnect,train_model_dir,top):
    # sechquery
    data_path = dbconnect
    simstfidf, simlda = get_simst_topic(train_model_dir, query)
    simlda_indexs = []
    for v in get_topmax(simlda,top):
        simlda_indexs.append(v[0])
    return  simlda_indexs


from flask import Flask,request
app = Flask(__name__)
@app.route("/getSmtsTopic", methods=['Get', 'POST']) # curl http://127.0.0.1:8333/getSmtsTopic?query=salary+job+paycheck+cython+incomes+budgeted+ratios\&topnum=4
def getSmtsTopic():                                     # http://127.0.0.1:8333/getSmtsTopic?query=A healthy diet is very important for people. It contains some fat, some  fibre, a little salt and so on. The Chinese diet is considered to be the healthies  in the world. It contains a lot of fruit and vegetables. It is rich in fibre and low in sugar and fat. As a result, a lot of Chinese have healthy white teeth. So I am very happy to be Chinses . People in west eat too much fat and sugar. They like to eat butter, cream, chocolate, sweet and so on, which contain a lot of sugar. As a result, they put on weight easily. Many of them become very fat and some have bad teeth. The worset  thing is that many westerners die in an early age from the heart illness. So I thought   eat more fruit and vegetables, not to ear  too much sugar and fat is a real healthy diet.\&topnum=4
    query = ''
    topnum_c = '4'
    dic = []
    if request.method == 'POST':
        query = request.form['query']
        topnum_c = request.form['topnum']
    else:
        query = request.args.get('query')
        topnum_c = request.args.get('topnum')
    topnum = int(topnum_c)
    #服务器结果
    # simlda_indexs = get_query_simst(query, data_path, train_model_dir, topnum)
    # sech_data = search_topdata(simlda_indexs, data_path)
    # sum_all_list = []
    # for sets in sech_data:
    #     text = sets.replace('\n', '').replace('\r', '')
    #     sum_text = ''
    #     s_sum_list = extract_top_sentences(text,3)
    #     for s_sum in s_sum_list:
    #         sum_text = '......'+s_sum+'......'
    #     sum_all_list.append(sum_text)
    # result = json.dumps(sech_data, encoding="UTF-8", ensure_ascii=False, indent=4)
    #本地联调
    sum_all_list = ['...And the other important thing is the foods can make many westerners die at an early from heart illness .......I think this diet list is good for people: everyday eat two fruits, eat more vegetables in dinner .......But in the west, people love eat  hamburgers, potato crisps, potato chips, butter, and chocolate.......But if people eat more these  food, would put on weight very easily ....', '...As a result, we now that a healthy diet should contain lots of fruit, green vegetables, because these  thing are low in sugar and fat but rich in fibre.......Not to eat too much potato crisps, potato chips, butter cream and chocolate cakes, soft drinks, sweet.......People in the western eat too much fat, sugar, salt and do not take enough exercise.......The healthy food should contain some fat, some fibre, a little salt and so on....', "...Potato crisps, potato chips, butter, soft drinks, cakes and so on are the thinks which can make them fat .......In forgin  countries, people don't eat such healthy food.......People eat less sugar than many countries in the world .......There's difference between Chinese and Western diet....", "...It's foods which contain some fat, some fibre, a little salt and so on.......Because Chinese eat less sugar than many other countries in the world .......The Chinese diet is considered to be the healthiest in the world.......But people in the western world do not eat such healthy foods....", "...But there's one thing I know is ture : the Chinese food is considered to be healthiest diet in the world.......Chinese diet contains lots of fruit and vegetables, so it's rich in fibre and low in sugar and fat.......In some parts of Britain, one preson  in ten, by the age of thirty, has no teeth left!......Because of this, many of them not only had teeth, but also the heart illness....", "...The Chinese diet contains a lot of fruit and green vegetable, it's considered to be the healthiest in the world.......As a result, many westerners diet  at heart illness and most of them have weight problem.......A healthy diet should contain some fat, some fibre, a little sugar and so on.......So, if you want to have a healthy body , you must have a healthy diet....", "...The western world , don't eat too much fat and sugar and don't take enough exercise .......You can do more exercise, don't eat much more chocolate, sugar and icecream.......If you are thin, you can eat hamburgar , chocolater , sugar and so on.......The Chinese diet is considered to be the healthiest in the world....", '...Severy  years ago, when we are first  , to study English, we must read a word, just a word again and again, some words probably are read for several hundreds more  .......All of us can have a sense  that when we do something again and again, we then can do it etter and faster.......For another example, in the factory, workers are busy to fit  with radio.......Seven years later, we now master much knoledge  about English....', "...From child to adult, we study in school constantly, only know theroy  knowledge from the book  , that is not enough, we must combine  practise  in society, so that, we can learn  knowledge really  .......We can make society and marketing investigation, it can let us know that  social department's struture  and running, and through the information, we can see the system and requirement of society.......I would  try my best to get to know about society and the world outside the campus, learn more news from newspaper, radio, book, medium etc......., and take part in the practice in society, and study hard to serve the society in  future...."]
    # result = json.dumps(sum_all_list, encoding="UTF-8", ensure_ascii=False, indent=4)
    result = json.dumps(sum_all_list,  ensure_ascii=False, indent=4)
    return result

app.run(host='0.0.0.0',port=8333)

