# -*- coding: utf-8 -*-
import os, sys
import json
import math
from gensim import corpora,models,similarities,utils
from IntelligentAssistantWriting.demol.utils import  *
from IntelligentAssistantWriting.demo.topic_demo import filter_char

train_model_dir = 'E:/temp/train_temp'
data_path = 'E:/myworkspace/data/train_data'

# 载入字典
dictionary = corpora.Dictionary.load(os.path.join(train_model_dir, "all.dic"))
# 载入TFIDF模型和索引
tfidfModel = models.TfidfModel.load(os.path.join(train_model_dir, "allTFIDF.mdl"))
indexTfidf = similarities.MatrixSimilarity.load(os.path.join(train_model_dir, "allTFIDF.idx"))
# 载入LDA模型和索引
ldaModel = models.LdaModel.load(os.path.join(train_model_dir, "allLDA50Topic.mdl"))
indexLDA = similarities.MatrixSimilarity.load(os.path.join(train_model_dir, "allLDA50Topic.idx"))

def filter_char(content):
    if ('.txt' in content) or (')' in content) or ('(' in content) or ('<' in content)or ('>' in content):
        content = ''
    else:
      content = content.replace(',','').replace('.','').replace(';','').replace('<','').replace('>','')
    return content

def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down

def get_simst_topic(topic,article,train_model_dir):
    # query就是测试数据，先切词
    # query = "this a happy day,I am so happy"
    content = (article.lower()).split(' ')
    topic = (topic.lower()).split(' ')
    # print content
    content_bow = dictionary.doc2bow(filter(lambda x: len(x) > 0, map(filter_char, content)))
    topic_bow = dictionary.doc2bow(filter(lambda x: len(x) > 0, map(filter_char, topic)))
    # print query_bow
    # 使用TFIDF模型向量化
    tfidfvect = tfidfModel[content_bow]
    tfidfvect_t = tfidfModel[topic_bow]
    # print tfidfvect
    # 然后LDA向量化，因为我们训练时的LDA是在TFIDF基础上做的，所以用itidfvect再向量化一次
    ldavec = ldaModel[tfidfvect]
    ldavec_t = ldaModel[tfidfvect_t]

    # TFIDF相似性
    # simstfidf = indexTfidf[tfidfvect]
    # LDA相似性
    # simlda = indexLDA[ldavec]

    # print ldavec
    vecArt = [c[1] for c in ldavec]
    vecTopic = [c[1] for c in ldavec_t]

    # LDA相似性
    simlda = cos_dist(vecArt,vecTopic)

    # print simstfidf
    # print simlda

    # print max(simlda.items(),key=lambda x:x[1])
    # print  simlda.max()
    # print 'ok'
    return  simlda

def get_simst_content(train_model_dir,query):
    # 载入字典
    dictionary = corpora.Dictionary.load(os.path.join(train_model_dir, "all.dic"))
    # 载入TFIDF模型和索引
    tfidfModel = models.TfidfModel.load(os.path.join(train_model_dir, "allTFIDF.mdl"))
    indexTfidf = similarities.MatrixSimilarity.load(os.path.join(train_model_dir, "allTFIDF.idx"))
    # 载入LDA模型和索引
    ldaModel = models.LdaModel.load(os.path.join(train_model_dir, "allLDA50Topic.mdl"))
    indexLDA = similarities.MatrixSimilarity.load(os.path.join(train_model_dir, "allLDA50Topic.idx"))

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

def get_topic_compare(topic,article,dbconnect,train_model_dir):
    # sechquery
    data_path = dbconnect
    simlda = get_simst_topic(topic,article,train_model_dir)
    return  simlda


from flask import Flask,request
app = Flask(__name__)
@app.route("/getSmtsTopicCompare", methods=['Get', 'POST'])
# curl http://127.0.0.1:8334/getSmtsTopicCompare?topic=salary+job+paycheck+cython+incomes+budgeted+ratios\&article=salary+job+paycheck+cython+incomes+budgeted+ratios+budgeted+ratios
def getSmtsTopicCompare():
    topic = ''
    article = ''
    dic = []
    if request.method == 'POST':
        topic = request.form['topic']
        article = request.form['article']
    else:
        topic = request.args.get('topic')
        article = request.args.get('article')

    simlda = get_topic_compare(topic, article, data_path, train_model_dir)
    result = json.dumps(simlda, encoding="UTF-8", ensure_ascii=False, indent=4)
    return result

app.run(host='0.0.0.0',port=8334)

