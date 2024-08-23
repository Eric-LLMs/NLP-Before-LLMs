# -*- coding: utf-8 -*-
import os,sys
# from IntelligentAssistantWriting.conf.params import *
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext
# conf = create_params_calssification()
# train_file = os.path.join(conf.dir_root,"train_data_fastText")
# test_file = os.path.join(conf.dir_root,"test_data_fastText")
# model_path = os.path.join(conf.dir_root,"fastText.model")
#训练模型
train_file = os.path.join("train_data_fastText")
test_file = os.path.join("test_data_fastText")
model_path = os.path.join("fastText.model")
classifier = fasttext.supervised(train_file,model_path,label_prefix="__label__")
result = classifier.test(test_file)
print (result.precision)
print (result.recall)