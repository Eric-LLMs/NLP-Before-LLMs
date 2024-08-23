#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def add_layer(input_data,input_size,output_size,activation_function = None,keep_r=0.8):
    Weight = tf.Variable(tf.random_normal([input_size,output_size]))
    biase  = tf.Variable(tf.Variable(tf.zeros([1,output_size])+0.1))
    Wx_plus_b = tf.matmul(input_data,Weight)+biase
    Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_r)
    if activation_function is None:
       output_data = Wx_plus_b
    else:
       output_data = activation_function(Wx_plus_b)
    return  output_data

x_holder = tf.placeholder(tf.float32,[None,200],name = 'x_input')
y_holder = tf.placeholder(tf.float32,[None,4],name = 'y_input')
k_holder = tf.placeholder(tf.float32,[None,4],name = 'keep_holder')
# l1 = add_layer(x_holder,200,500,activation_function=tf.nn.softmax)
# l2 = add_layer(l1,5,5,activation_function= tf.nn.relu)
# y_pre = add_layer(l1,500,4,activation_function=None)
y_pre = add_layer(x_holder,200,4,activation_function=tf.nn.softmax,keep_r=1)


# ***************************
from IntelligentAssistantWriting.tf_demo.pre_data import *
data_path = '/home/enhui/myworkspace/data/demo_data'
# data_path = '/home/enhui/myworkspace/data/demo_data_test'
log_dir = '/home/enhui/temp/train_temp'
# data_path = os.path.join(log_dir, data_file_name)
pre_info_path = os.path.join(log_dir, 'Pre.info')
pre_data_path = os.path.join(log_dir, 'Pre.data')
train_size = 0.9
try:
    # pre_data(data_path, pre_data_path, pre_info_path)
    print('pre data has been finished')
except Exception as Error:
    print ('Error')
try:
    # 加载保存的数据
    preInfo = loadPreInfo(pre_info_path)
    preData = loadPreData(pre_data_path)
    # 训练并保存模型
    # result = stm_lstm_train(preInfo, preData, model_path, weight_path)
    (train_X, train_y), (test_X, test_y) = data_to_vector(preInfo, preData, train_size)

    print ('X_train :')
    print  (train_X.shape)

    print ('X_test :')
    print  (test_X.shape)

    nb_classes = len(corpus_tags)
    train_y_toC = to_categorical(train_y, nb_classes)
    test_y_toC = to_categorical(test_y, nb_classes)

    print ('Y_train_toC :')
    print  (train_y_toC.shape)

    print ('Y_test_toC :')
    print  (test_y_toC.shape)
    x_data = train_X
    y_data = train_y_toC
    x_test_data = test_X
    y_test_data = test_y_toC
    # print (train_X, train_y_toC), (test_X, test_y_toC)
except Exception as Error:
    print ('Error')


sess = tf.Session()

def test_accu(v_x,v_y):
    global  pre
    y = sess.run(y_pre, feed_dict={x_holder: v_x})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(v_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    accu_result = sess.run(accuracy, feed_dict={x_holder: v_x, y_holder: v_y})
    return accu_result

# loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_holder-y_pre),reduction_indices = [1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(tf.clip_by_value(y_pre,1e-10,1.0))))
train = tf.train.AdadeltaOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()

epochs = 100
batch_size = 50
sess.run(init)
for j in range(epochs):
    epoch_loss = 0
    accu_result = 0.00
    i = 0
    while(i<= len(x_data)):
        start = i
        end = i+batch_size
        x_batch = np.array(x_data[start:end])
        y_batch =  np.array(y_data[start:end])
        _,batch_loss = sess.run([train,cross_entropy],feed_dict={x_holder:x_batch,y_holder:y_batch})
        epoch_loss+=batch_loss
        i += batch_size
    print (j,epoch_loss)
    print  (test_accu(x_test_data, y_test_data))


sess.close()
