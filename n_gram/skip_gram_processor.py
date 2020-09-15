# -*- coding: utf-8 -*-

"""
@File: skip_gram_processor.py
@Author: eric
@Time: 09/23/2020
"""

from gensim.models import word2vec


class SKPGram():
    def __init__(self, model_path):
        self.name = 'SKPGram'
        self.model_path = model_path
        self.model = word2vec.Word2Vec.load(self.model_path)

    def get_socre(self, cur_segment, pos, window=1):
        candi_char = cur_segment[pos]
        confidence_score = 0.00
        len_cur_segment = len(cur_segment)
        if len_cur_segment > 1:
            for i in range(max(0, pos - window), min(pos + window, len_cur_segment - 1) + 1):
                cur_char = cur_segment[i]
                if i != pos and cur_char != '[blank]' and i >= 0 and (i <= len_cur_segment - 1):
                    try:
                        confidence_score += max(0, self.model.similarity(cur_char, candi_char))
                    except Exception as e:
                        continue
            confidence_score = confidence_score / float(len_cur_segment - 1)
        return confidence_score

    def train(self, train_file):
        sentences = word2vec.Text8Corpus(train_file)
        model = word2vec.Word2Vec(sentences, min_count=1, window=3, size=128)
        model.save(self.model_path)