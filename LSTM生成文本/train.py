import tensorflow as tf
import numpy as np
import time

from model import LSTM
from file_parse import get_batches,vocab,encode

#-----------------超参数-------------------

learning_rate = 0.01
traing_iters = 5000
n_hidden_units = 128
batch_size = 64
num_steps = 50
num_class = len(encode)
epoches = 20
save_freq = 200
keep_prob = 0.5
#-----------------超参数-------------------


model = LSTM(len(vocab),batch_size,num_steps=num_steps , lr=learning_rate)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep= 200)

with tf.Session() as sess:
    sess.run(init)

    for ep in range(epoches):
        new_state = sess.run(model.init_state) #每次跑完一个epoch，都初始化一下状态
        # print(np.shape(new_state))
        counter = 0
        for x,y in get_batches(encode,batch_size,num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs : x,   model.label :y,
                    model.keep_prob:keep_prob, model.init_state : new_state }
            batch_loss,new_state,_ = sess.run([model.loss,model.final_state,model.optimizer],
                                              feed_dict= feed)
            end = time.time()

            print('Epoch : {} / {}...'.format(ep+1,epoches),'  Training steps :{}'.format(counter),
            '  Training loss : {:.4f}'.format(batch_loss),'  {:.4f} sec/batch'.format(end-start))
            if  counter % save_freq == 0 :
                saver.save(sess,'checkpoints/iter{}.ckpt'.format(counter))
                print('\n--------save ok!---------\n')

        saver.save(sess,'checkpoints/end_iter{}.ckpt'.format(counter))

if __name__ == '__main__':
    print('this is the main code !')