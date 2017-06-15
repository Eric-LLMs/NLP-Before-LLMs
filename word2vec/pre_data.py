import os,sys
from gensim.models import word2vec

def get_sentences(dir_data):
    sentences = []
    for file in os.listdir(dir_data):
        file_path = os.path.join(dir_data, file)
        sentences.append(word2vec.Text8Corpus(file_path))
        # fopen = open(file_path, 'r')  # r 代表read
        # for line in fopen:
        #     sentences.append(line)
        # fopen.close()
    return sentences