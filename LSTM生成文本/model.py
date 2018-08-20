import tensorflow as tf
import numpy as np
from file_parse import vocab
from file_parse import vocab_to_int,int_to_vocab,vocab

num_class = len(vocab)
n_hidden_units = 128
keep_prob = 0.5
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
                 lr = 0.01 ,grad_clip=5,w = weights,bia = bias,is_predicting = False,predic_input_num =1):
        if is_predicting == True:
            batch_size = predic_input_num #如果是要预测的话，batch_size=predic_input_num
            num_steps = 1 # 步数设为1，每输入一句话，就预测一句话
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
        # print('这里是lstm函数',self.prediction.shape)
        # print(self.label.shape)
        self.loss = calculate_loss(self.prediction,self.label,num_class)

        self.optimizer = build_optimizer(self.loss,lr,grad_clip = grad_clip)

#定义一个生成文字的函数，当模型训练好了就调用这个函数
def generate_novel(head_words,out_amount):
    directory = 'E:/py_pro/poem_gen/checkpoints/'
    print('打开已经训练好的模型...')
    input_words = [c for c in head_words]#将输入的句子划分成字？
    input_encode = list(map(lambda x:vocab_to_int[x],input_words)) #将输入的汉字映射成数字
    input_encode = np.reshape(input_encode,[np.shape(input_encode)[0],1])
    # print(input_encode)
    novel_model = LSTM(num_class=len(vocab),is_predicting=True,predic_input_num=np.shape(input_encode)[0])
    out = [] #定义一个空列表,用来存放生成的汉字
    tempt =[]
    predict_saver = tf.train.Saver()
    with tf.Session() as sess1:
        print('开始加载图...')
        # my_saver = tf.train.import_meta_graph(directory+'end_iter.meta')#加载图结构
        # my_saver.restore(sess1,tf.train.latest_checkpoint(location))#取最后一个保存点
        checkpoint = tf.train.latest_checkpoint(directory)
        predict_saver.restore(sess1,checkpoint)
        print('模型加载成功 !')
        new_state = sess1.run(novel_model.init_state)#将模型训练得到最后一个状态作为训练的初始状态
        feed ={ novel_model.inputs : input_encode,
                novel_model.keep_prob:keep_prob,
                novel_model.init_state:new_state}
        #模型预测输出，并输出最后的状态，作为下一次的初始状态,这里根据输入预测出了第一句话
        pred,new_state = sess1.run([novel_model.prediction,novel_model.final_state],feed_dict=feed)
        #这里的prediction是一个( time_steps*batch_size , 3881)形状的矩阵
        # print('热独码解码后的形状为：',np.argwhere(pred == 1))
        #这里是获取每一行为1的列索引，也就是进行onehot解码
        # pred_index = np.argwhere(pred == 1)[:][1]#np.argwhere返回的是一个二维数组（a，b）,这里的b就是输出汉字对应的数字
        pred_index = np.argmax(pred,axis=1)
        # print(pred_index.shape)
        for c in pred_index:
            tempt.append(int_to_vocab[c])
        out.append(''.join(tempt))
        tempt.clear()

        pred = pred_index.reshape([np.shape(pred_index)[0],1])
        # print(pred.shape)
        for i in range(out_amount-1):
            feed = {novel_model.inputs: pred,
                    novel_model.keep_prob: keep_prob,
                    novel_model.init_state: new_state}

            prediction, new_state = sess1.run([novel_model.prediction, novel_model.final_state], feed_dict=feed)
            # print(prediction.shape)
            # prediction是一个( time_steps*batch_size , 3881)形状的矩阵
            # 这里的b就是输出汉字对应的数字，onehot解码
            pred_index = np.argmax(prediction, axis=1)
            for c in pred_index:
                tempt.append(int_to_vocab[c])
            out.append(''.join(tempt))
            tempt.clear()
            pred = pred_index.reshape([np.shape(pred_index)[0], 1])

    return out #预测完毕，返回

if __name__ == '__main__':
    print('success!')
    model = LSTM(len(vocab),batch_size=64,num_steps=50 , lr=0.01)



