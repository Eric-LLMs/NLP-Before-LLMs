# -*- coding: utf-8 -*-

"""
@File: corrector.py
@Author: eric
@Time: 23/09/2020
"""

import math
import numpy as np

from src.decoder import *
from src.detector import *
from src.utils.utils import cut_sentences
from src.ngram_processor import *
from src.skip_gram_processor import *
from src.config import *
from src.utils.logger import get_logger

__dir__ = os.path.dirname(os.path.dirname(__file__))
sys.path.append(__dir__)

class Corrector:
    def __init__(self, ngram2, skpgram, threshold):
        self.name = 'corrector'
        self.check_dic = {}
        self.confusion_dic = {}
        self.config = config(
            root=__dir__,
            n_gram_model_path=ngram2,
            skp_gram_model_path=skpgram,
            threshold=threshold,
        )
        self.decoder = Decoder()
        self.detector = Detector()
        self.ngram_processor = NGram(self.config.n_gram_model_path)
        self.skp_gram_processor = SKPGram(self.config.skp_gram_model_path)
        # self.logger = get_logger(self.name, os.path.join(self.config.logger_dir, 'ocr_coreector_log.log'), 'ERROR')
        self.logger = get_logger(self.name)

    # 加载混淆词典
    def load_confusion_dic(self,):
        # 配置文件写词典路径
        pass

    def smooth(self,x):
        y = np.tan(np.pi / 3 * (x ** 7))
        return y

    def rewrite_text_char_by_char(self, text):
        """
         一个字一个字的修改
        :param text:
        :return:
        """
        self.decoder.load_file(text)
        cur = 'default'
        # 预测序列
        pd_text_list = []
        # 预测序列概率
        pd_prob_list = []
        words = set()
        char_index_detail = {}
        pro_index_detail = {}
        skip_chars = ['[blank]']
        for pt, pd in zip(self.decoder.pd_text, self.decoder.pd_prob):
            t = pt[0]
            p = pd[0]

            for item in pt[1:]:
                words.add(item)

            # 中间没有[blank] 连续合并成一个，中间有[blank]重复保留
            if cur == t:
                continue
            cur = t

            if t not in skip_chars:
                pd_text_list.append(t)
                pd_prob_list.append(p)
                # t_index = pd_text_list.index(t) #重复字取最前面的索引，出错
                t_index = len(pd_text_list) - 1
                p_index = len(pd_prob_list) - 1
                char_index_detail[t_index] = pt  # 候选字列表
                pro_index_detail[p_index] = pd  # 候选字概率

        pd_text_str = ''.join(pd_text_list)

        # correction_sque 逐字纠错后的序列，初始值为pd_text_str
        correction_sque = pd_text_str
        pre_seq_list = pd_text_list
        threshold = self.config.threshold

        info_detail = {}
        for prob in pd_prob_list:
            # 在文本中的位置
            i_prob = pd_prob_list.index(prob)
            if float(prob) < threshold:
                max_lm = -float('inf')
                candidate_list = char_index_detail[i_prob]
                candidate_list_probs = pro_index_detail[i_prob]

                # 文本分句子，并返回句子位置和长度
                pd_text_str_sen_pos_dic, pd_text_str_sentents = cut_sentences(correction_sque)

                # 当前要修改的词，所在句子（文本有多个句子）中的位置
                cur_segment_candi_char_index = 0
                cur_segment = []
                for sent_index in pd_text_str_sen_pos_dic:
                    [p, l] = pd_text_str_sen_pos_dic[sent_index]
                    if i_prob >= p and i_prob < (p + l):
                        cur_segment = [i for i in pd_text_str_sentents[sent_index]]
                        cur_segment_candi_char_index = i_prob - p

                # # 人名不做修改
                # if self.detector.check_name(self.decoder.gt_text, i_prob):
                #     continue
                #
                # # 检错正确，不做修改
                # if self.detector.check(pre_seq_list, i_prob):
                #     continue

                if len(cur_segment) < 2:  # 只修改大于等于2的term 或者 短句
                    continue

                for candi_char, candi_probs in zip(candidate_list, candidate_list_probs):
                    if candi_char in skip_chars :
                        continue
                    cur_segment[cur_segment_candi_char_index] = candi_char
                    pro_ngram, perplexity = self.ngram_processor.get_socre(cur_segment)
                    skip_gram_socre = self.skp_gram_processor.get_socre(cur_segment,
                                                                        cur_segment_candi_char_index, window=2)
                    ocr_smooth = self.smooth(float(candi_probs))
                    # 单字与相邻字的语义相关性用skip_gram_socre，整个句子靠pro_ngram与perplexity
                    score = (2 + 80 * skip_gram_socre) * math.pow(10, 2 * float(pro_ngram))* ocr_smooth / perplexity
                    # score = ((2 + 80 * skip_gram_socre) * math.pow(10, 2 * float(pro_ngram)) / perplexity)

                    info_detail_key = "%s_%s" % (candi_char, str(len(info_detail)))
                    info_detail[info_detail_key] = (
                            "{pd_text_list : %s, pd_prob_list : %s, cur_segment : %s, score : %s, pro_ngram : %s, perplexity : %s,  skip_gram_socre : %s, candi_probs : %s, ocr_smooth : %s }"
                            % (pd_text_list, pd_prob_list, cur_segment, score, pro_ngram, perplexity, skip_gram_socre, candi_probs,
                               ocr_smooth))

                    if score > max_lm:
                        max_lm = score
                        pre_seq_list[i_prob] = candi_char
                        correction_sque = ''.join(pre_seq_list)  # 修改后OCR输出结果

        self.logger.debug("%s,%s,%s" % (pd_text_str,correction_sque, str(info_detail)))
        return correction_sque
