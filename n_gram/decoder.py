# -*- coding: utf-8 -*-

"""
@File: decoder.py
@Author: eric
@Time: 23/09/2020
"""

import json

class Decoder:
    def __init__(self):
        self.name = 'decoder'
        self.info = None  # 输入信息
        self.gt_text = None  # 标注文本，用于测评
        self.pd_text = None  # OCR char top5
        self.pd_prob = None  # OCR char top5 概率
        self.words = set()  # OCR 召回所有char列表
        self.pd_text_list = []  # OCR 解码，预测文本序列
        self.pd_prob_list = []  # OCR解码，预测文本概率
        self.char_index_detail = {}  # 候选字列表
        self.pro_index_detail = {}  # 候选字概率

    def load_file(self, file):
        self.clear()
        self.info = json.loads(file)
        if 'img_path' in self.info:
            self.img_path = self.info['img_path']
        if 'gt_text' in self.info:
            self.gt_text = self.info['gt_text']
        self.pd_prob = self.info['pd_prob']
        self.pd_text = self.info['pd_text']
        self.decoder_file()

    def clear(self):
        self.info = None
        self.gt_text = None
        self.pd_text = None
        self.pd_prob = None
        self.words = set()
        self.pd_text_list = []
        self.pd_prob_list = []
        self.char_index_detail = {}
        self.pro_index_detail = {}

    def decoder_file(self):
        # [blank] 去掉，相邻重复的合并
        cur = 'default'
        for pt, pd in zip(self.pd_text, self.pd_prob):
            t = pt[0]
            p = pd[0]

            for item in pt[1:]:
                self.words.add(item)

            # 中间没有[blank] 连续合并成一个，中间有[blank]重复保留
            if cur == t:
                continue
            cur = t

            if t != '[blank]':
                self.pd_text_list.append(t)
                self.pd_prob_list.append(p)
                # t_index = self.pd_text_list.index(t) # 重复字取最前面的索引，出错
                t_index = len(self.pd_text_list) - 1
                p_index = len(self.pd_prob_list) - 1
                self.char_index_detail[t_index] = pt  # 候选字列表
                self.pro_index_detail[p_index] = pd  # 候选字概率