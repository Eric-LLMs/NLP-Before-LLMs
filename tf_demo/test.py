import tensorflow as tf
import numpy as np

def add_layer(input_data,input_size,output_size,n_layer,activation_function = None):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('W'):
            Weight = tf.Variable(tf.random_normal([input_size,output_size]))
            tf.histogram_summary(layer_name+'weight',Weight)
        with tf.name_scope('biase'):
            biase  = tf.Variable(tf.Variable(tf.zeros([1,output_size])+0.1))
            tf.histogram_summary(layer_name+'/biase',biase)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input_data,Weight)+biase
    if activation_function is None:
       output_data = Wx_plus_b
    else:
       output_data = activation_function(Wx_plus_b)
    tf.histogram_summary(layer_name + '/output_data', output_data)
    return  output_data

x_data = np.linspace(-1,1.,500,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = x_data - 0.7 + noise

with tf.name_scope('input'):
    x_holder = tf.placeholder(tf.float32,[None,1],name = 'x_input')
    y_holder = tf.placeholder(tf.float32,[None,1],name = 'y_input')

l1 = add_layer(x_holder,1,20,n_layer=1,activation_function=tf.nn.relu)
# l2 = add_layer(l1,5,5,activation_function= tf.nn.relu)
y_pre = add_layer(l1,20,1,n_layer='out_put',activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_holder-y_pre),reduction_indices = [1]))
    tf.scalar_summary('loss',loss)
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('/home/enhui/temp/log',sess.graph)
    sess.run(init)
    for i in range(1000):
         sess.run(train,feed_dict={x_holder:x_data,y_holder:y_data})
         if i%200==0:
            # print(sess.run(loss, feed_dict={x_holder: x_data, y_holder: y_data}))
            result = sess.run(merged, feed_dict={x_holder: x_data, y_holder: y_data})
            writer.add_summary(result,100)
            print(sess.run((y_pre-y_data)/y_data, feed_dict={x_holder: x_data, y_holder: y_data}))