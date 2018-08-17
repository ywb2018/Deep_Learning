#coding=utf-8
import os
import tensorflow as tf
import numpy as np

direction = 'E:/py_pro/poem_gen/苍天万道.txt'
locations = r'E:\py_pro\poem_gen\测试.txt'

#去空行函数
def clearBlankline(file):
    file2 = open(r'E:\py_pro\poem_gen\新小说.txt','w')
    with open(file) as f:
        for line in f.readlines():
            if line =='\n':
                line=line.strip()
            file2.write(line)
    file2.close()
clearBlankline(direction)#去空行生成新文档


#-----------将小说进行编码----------------------------
##思来想去，觉得可以保留小说中的换行符，这样生成的小说也可以自己换行
with open(r'E:\py_pro\poem_gen\新小说.txt','r') as f:
    text = f.read().strip()
vocab = sorted(set(text))#将集合排序


vocab_to_int = {c:i for i,c in enumerate(vocab)} #将字符列表构建字典
int_to_vocab = dict(enumerate(vocab))  #构建与上相反的字典
encode = np.array([vocab_to_int[c] for c in text],dtype=np.int32) #将小说里的所有文字进行编码
#-----------将小说进行编码----------------------------




#------------生成Mini-baches-----------------------------------
def get_batches(arr,n_seqs,n_steps): #这里n_seqs 视作batch_size，n_steps就是时间序列数，input_size = 1

    slice = n_seqs * n_steps
    n_batches = len(arr)//slice #看下有多少个slice
    arr = arr[: n_batches * slice] #取整
    arr = arr.reshape((n_seqs,-1)) #reshape 成(n_seqs , n_steps * n_batches)的形状，是为了后面取下一个sequence值作为label值方便
    for n in range(0, arr.shape[1], n_steps):#总共有n_batches个循环
        x = arr[:,n:n + n_steps]  #每次取一个slice
        y = np.zeros_like(x)
        #每个x[i] sequence对应的label为 x[i+1],y的最后一个sequence取了输入的第一个sequence(这里是一个问题，是可以优化的)
        y[:,:-1],y[:,-1] = x[:,1:],x[:,0]
        yield x,y



if __name__ == '__main__':
    print('\n This is a main code ...')
    print('汉字集合的长度为', len(vocab))
