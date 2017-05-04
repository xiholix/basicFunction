# -*-coding:utf8-*-
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
#
#
'''
@version: ??
@author: xiholix
@contact: x123872842@163.com
@software: PyCharm
@file: embeddingLookup.py
@time: 17-5-4 上午9:31
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import  numpy as np


def test_embedding_lookup():
    a = np.arange(8).reshape(2,4)
    b = np.arange(8,12).reshape(1,4)
    c = np.arange(12, 20).reshape(2,4)
    print(a)
    print(b)
    print(c)

    a = tf.Variable(a)
    b = tf.Variable(b)
    c = tf.Variable(c)

    t = tf.nn.embedding_lookup([a,b,c], ids=[0,1,2,3])
    # 此处如果ids=[0,1,2,3]不会报错，因为此时并没有发现b比c要少一行，程序能够正常的执行，但是如果出现参数4了，因为
    # 程序的partition要求在无法进行均匀切分时，前面的(max_id+1)%len(params)个param的切分可以多一个。在此例子中
    # 正确的id应该是params中的第一元素的id为[0,3], 第二元素的id应该为[1,4], 第三个元素的id应该为[2]。所以正确的param
    # 应该是（a,c,b)或者(c,a,b),总之b应该放在最后面
    # 本例的运算结果为：
    '''
    [[ 0  1  2  3]
    [ 8  9 10 11]
    [12 13 14 15]
    [ 4  5  6  7]]
    '''
    #但是本例中的[a,b,c]顺序其实是错误的

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    m = sess.run(t)
    print(m)


def test_lookup_sparse():
    a = np.arange(8).reshape(2, 4)
    b = np.arange(8, 16).reshape(2, 4)
    c = np.arange(12, 20).reshape(2, 4)

    print(a)
    print(b)
    print(c)

    a = tf.Variable(a, dtype=tf.float32)
    b = tf.Variable(b, dtype=tf.float32)
    c = tf.Variable(c, dtype=tf.float32)

    idx = tf.SparseTensor(indices=[[0,0], [0,2], [1,0], [1, 1]], values=[1,2,2,0], dense_shape=(2,3))
    result = tf.nn.embedding_lookup_sparse((a,c,b), idx, None, combiner="sum")

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    r = sess.run(result)
    print(r)
    '''
    根据程序的测试结果来看，这里的params的结合方式并不是成为一个逻辑大tensor，而是直接变成一个大的tensor，在该tensor的在第0维扩张
    '''
    '''
    a,b,c 为：[[0 1 2 3]
 [4 5 6 7]]

[[ 8  9 10 11]
 [12 13 14 15]]

[[12 13 14 15]
 [16 17 18 19]]

 在实现中好像将它们结合成一个大的tensor了，而不是使用partition,即实现的结果是
 [[[0 1 2 3]
 [4 5 6 7]]
[[ 8  9 10 11]
 [12 13 14 15]]
[[12 13 14 15]
 [16 17 18 19]]
 ]
 最后的结果为：
 [[[ 20.  22.  24.  26.]
  [ 28.  30.  32.  34.]]

 [[  8.  10.  12.  14.]
  [ 16.  18.  20.  22.]]]

    '''

if __name__ == "__main__":
    test_lookup_sparse()