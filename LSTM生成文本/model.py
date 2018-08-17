import tensorflow as tf
import numpy as np
from file_parse import vocab
# from train import n_hidden_units

num_class = len(vocab)
n_hidden_units = 128
# x.shape = (n_seqs,n_steps),这里 n_seqs 被视作 batch_size，目前是input_size = 1
weights = {
    'in' : tf.Variable(tf.random_normal([num_class,n_hidden_units]),tf.float32),
    'out': tf.Variable(tf.random_normal([n_hidden_units,num_class]),tf.float32)
}
bias = {
    'in' :tf.Variable(tf.random_normal([n_hidden_units]),tf.float32),
    'out':tf.Variable(tf.random_normal([num_class]),tf.float32)
}


def build_lstm(hidden_num,num_layer,batch_size):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units= hidden_num,forget_bias=1.0,state_is_tuple=True)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layer) #多层lstm
    init_state = multi_cell.zero_state(batch_size,dtype=tf.float32)
    # print(tf.Session().run())
    return multi_cell,init_state

def calculate_loss(pred,label,class_num):
    y_hot= tf.reshape(tf.one_hot(label,class_num),[-1,class_num])#将标签值热独编码
    # print('y_hot的形状是：',y_hot.shape)
    # print('predict的形状是：',pred.shape)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y_hot))
    return loss

def build_optimizer(loss, learning_rate, grad_clip):

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer


class LSTM:
    def __init__(self,num_class,batch_size =64,num_steps =50,hidden_units=128,layer_num = 2,
                 lr = 0.01 ,grad_clip=5,w = weights,bia = bias):

        #清除一些多余的图结构
        # tf.reset_default_graph()
        # 输入数据的input_size = 1, batch_size=64   num_steps=50
        self.inputs = tf.placeholder(tf.int32,[batch_size,num_steps],name='input')
        self.label  = tf.placeholder(tf.int32,[batch_size,num_steps],name='label')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')

        self.cell,self.init_state = build_lstm(hidden_units,layer_num,batch_size)#两层

        # 将(batch_size,n_steps) 输入的数字变成热独编码  one_hot之后的shape = (batch_size, n_steps, num_class)
        x_in = tf.one_hot(self.inputs,num_class)
        x_in = tf.reshape(x_in,[-1,num_class])#(64*50 , 3881)
        w_in = tf.Variable(tf.random_normal([3881,hidden_units]),tf.float32)
        x_in= tf.matmul(x_in,w_in)+bia['in']
        x_in  = tf.reshape(x_in,[-1,num_steps,hidden_units])
        # print(x_in.shape)
        # print(x_in.shape)
        # x_in = tf.reshape(x_in,[-1,num_steps,hidden_units])
        output,state = tf.nn.dynamic_rnn(self.cell,inputs= x_in,initial_state= self.init_state,time_major =False)
        self.final_state = state
        #这里加了一个神经网络到num_class的全连接层
        output = tf.reshape(output,[-1,hidden_units])
        self.prediction = tf.matmul(output,w['out'])+bia['out']

        # self.prediction = self.state[1][1]
        # print(self.final_state[1][1].shape)
        # print(output.shape)
        print('\n')
        # print(self.prediction.shape)
        # print(self.label.shape)
        self.loss = calculate_loss(self.prediction,self.label,num_class)

        self.optimizer = build_optimizer(self.loss,lr,grad_clip = grad_clip)


if __name__ == '__main__':
    print('success!')
    model = LSTM(len(vocab),batch_size=64,num_steps=50 , lr=0.01)



