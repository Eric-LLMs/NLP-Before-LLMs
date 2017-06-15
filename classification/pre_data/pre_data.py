import os,sys
import tensorflow as tf

from IntelligentAssistantWriting.conf.params import *
config = create_params_calssification()
file_source = os.path.join(config.dir_root,'book_data_less_1000_random.txt')
file_target = os.path.join(config.dir_root,'train_data_word2vec')

train_data_raw = os.path.join(config.dir_root,'features_lexileLevel_text_all')
train_data = os.path.join(config.dir_root,'features_lexileLevel_text_all_xgboost')
train_data_fastText = os.path.join(config.dir_root,'train_data_fastText')
test_data_fastText = os.path.join(config.dir_root,'test_data_fastText')

def get_files():
  op = open(file_target, 'a', encoding='UTF-8')
  for line in open(file_source):
    op.write(line.replace('\n','').replace('\r','')+'\n')
  op.close()

def get_train_data_xgboost():
    op = open(train_data, 'w', encoding='UTF-8')
    index = 0
    for line in open(train_data_raw,'r',encoding='UTF-8'):
        if index==0:
            op.write(line.replace('\n', '').replace('\r', '')+ '\n')
            index+=1
            continue
        items = line.replace('\n', '').replace('\r', '').split('\t')
        features = items[0]
        # labels = items[1].split(',')
        lexile_level = items[2]
        filename = items[3]
        text = items[4]
        label = 0
        if lexile_level == 'BR1' or label == 'BR2':  # BR 设为0级
            label = 0
        elif lexile_level == '':  # 默认设为0级
            label = 0
        elif lexile_level == '0L - 100L':
            label = 1
        elif lexile_level == '100L - 200L':
            label = 2
        elif lexile_level == '200L - 300L':
            label = 3
        elif lexile_level == '300L - 400L':
            label = 4
        elif lexile_level == '400L - 500L':
            label = 5
        elif lexile_level == '500L - 600L':
            label = 6
        elif lexile_level == '600L - 700L':
            label = 7
        elif lexile_level == '700L - 800L':
            label = 8
        elif lexile_level == '800L - 900L':
            label = 9
        elif lexile_level == '900L - 1000L':
            label = 10
        elif lexile_level == '1000L - 1100L':
            label = 11
        elif lexile_level == '1100L - 1200L':
            label = 12
        elif lexile_level == '1200L - 1300L':
            label = 13
        elif lexile_level == '1300L - 1400L':
            label = 14
        else:
            label = 15
        label = str(label)
        op.write("%s,%s,%s,INDEX%s"%(features,str(label),lexile_level,index) + '\n')
        index+=1
    op.close()

def get_train_data_textCNN():
    op = open(train_data, 'w', encoding='UTF-8')
    index = 0
    for line in open(train_data_raw,'r',encoding='UTF-8'):
        if index==0:
            op.write(line.replace('\n', '').replace('\r', '')+ '\n')
            index+=1
            continue
        items = line.replace('\n', '').replace('\r', '').split('\t')
        features = items[0]
        # labels = items[1].split(',')
        lexile_level = items[2]
        filename = items[3]
        text = items[4]
        label = 0
        if lexile_level == 'BR1' or label == 'BR2':  # BR 设为0级
            label = 0
        elif lexile_level == '':  # 默认设为0级
            label = 0
        elif lexile_level == '0L - 100L':
            label = 1
        elif lexile_level == '100L - 200L':
            label = 2
        elif lexile_level == '200L - 300L':
            label = 3
        elif lexile_level == '300L - 400L':
            label = 4
        elif lexile_level == '400L - 500L':
            label = 5
        elif lexile_level == '500L - 600L':
            label = 6
        elif lexile_level == '600L - 700L':
            label = 7
        elif lexile_level == '700L - 800L':
            label = 8
        elif lexile_level == '800L - 900L':
            label = 9
        elif lexile_level == '900L - 1000L':
            label = 10
        elif lexile_level == '1000L - 1100L':
            label = 11
        elif lexile_level == '1100L - 1200L':
            label = 12
        elif lexile_level == '1200L - 1300L':
            label = 13
        elif lexile_level == '1300L - 1400L':
            label = 14
        else:
            label = 15
        label = str(label)
        op.write("%s\t%s\t%s\t%s\t%s"%(features,str(label),lexile_level,filename,text) + '\n')
        index+=1
    op.close()

def get_train_data_fastText():
    op_train = open(train_data_fastText, 'w', encoding='UTF-8')
    op_test = open(test_data_fastText, 'w', encoding='UTF-8')
    index = 0
    for line in open(train_data_raw, 'r', encoding='UTF-8'):
        if index == 0:
            # op_train.write(line.replace('\n', '').replace('\r', '') + '\n')
            index += 1
            continue
        items = line.replace('\n', '').replace('\r', '').split('\t')
        features = items[0]
        ave_comp_tree = float(features.split(',')[0])
        if ave_comp_tree > 800 :
            continue
        label = items[1]
        lexile_level = items[2]
        filename = items[3]
        text = items[4]
        if index%5 ==0:
            op_test.write("%s\t__label__%s" % (text.replace('\n', '').replace('\r', ''), label) + '\n')
        else:
            op_train.write("%s\t__label__%s" % (text.replace('\n', '').replace('\r', ''),label) + '\n')
        index += 1
    op_train.close()
    op_test.close()
flags = tf.app.flags
############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')
FLAGS = tf.app.flags.FLAGS

def main(_):
    print(FLAGS.m_plus)
    print(FLAGS.m_minus)
    print(FLAGS.lambda_val)

import codecs
import nltk
def genet_bilstm_input():
    training_data_path = os.path.join(config.dir_root, 'train_test_data_fastText')
    zhihu_f = codecs.open(training_data_path, 'r', 'utf8')  # -zhihu4-only-title.txt
    lines = zhihu_f.readlines()
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for i, line in enumerate(lines):
        x, y = line.split('__label__')  # x='w17314 w5521 w7729 w767 w10147 w111'
        y = y.replace('\n', '')
        x_list = sentences = tokenizer.tokenize(x)
        x = '\t'.join(x_list) + '\t'
        x = x.replace("\t", ' EOS ').strip()

        if i < 5:
            print("x0:", x)  # get raw x
        # x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        # if i<5:
        #    print("x1:",x_) #
        x = x.split(" ")
if __name__ == '__main__':
    # tf.app.run()  #执行main函数
    # genet_bilstm_input()
    for start, end in zip(range(0, 168, 50),range(0, 168, 50)):
        print(start,end)