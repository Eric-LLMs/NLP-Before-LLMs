# -*- coding: utf-8 -*-

"""
@File: ngram_processor.py
@Author: eric
@Time: 23/09/2020
"""

import kenlm


class NGram:
    def __init__(self, model_path):
        self.name = 'NGram'
        self.model_path = model_path
        self.model = kenlm.Model(self.model_path)

    def get_socre(self, query, bos=False, eos=False):
        text = ' '.join(query)
        score = self.model.score(text, bos=False, eos=False)
        perplexity = self.model.perplexity(text)
        return score, perplexity