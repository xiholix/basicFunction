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
@file: testmultiply.py
@time: 17-5-7 上午9:09
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


def test_multiply():
    a = np.arange(0,6)
    b = np.ones(shape=(5,6) ,dtype=np.int64)
    x = tf.constant(5.0, shape=[5, 6])
    w = tf.constant([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    a = tf.Variable(a)
    b = tf.Variable(b)
    r = tf.multiply(b, a)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    r = sess.run(r)
    print(r)
    '''NOTE: tf.multiply supports broadcasting. More about broadcasting here'''


def test_batch_matrix_vector():
    a = np.arange(16, dtype=np.int32).reshape((2,2,4))
    b = np.arange(16, dtype=np.int32).reshape((2,4,2))
    r = tf.matmul(a,b)
    '''
    matmul的两个参数的shape必须匹配，不能一个是矩阵，另一个是向量
    '''
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    r = sess.run(r)
    print(r)

def test_get_shape_value():
    a = np.arange(16).reshape((4,4))

    b = tf.Variable(a).get_shape()[0]
    '''
    get_shape()的到的tensor_shape.Dimension组成的列表, 对其元素取.value得到的是int类型
    '''
    print(b)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    b = sess.run(b)
    print(b)


def test_batchvm():
    def batch_vm(v, m):
        shape = tf.shape(v)
        rank = shape.get_shape()[0].value
        #rank为shape,因为shape就是由v的每个维组成，所以他有多少维度，则shape的第0维的值就是多少
        print(rank)
        v = tf.expand_dims(v, rank)

        vm = tf.multiply(v, m)

        return tf.reduce_sum(vm, rank - 1)

    def batch_vm2(x, m):
        t = m.get_shape().as_list()
        print(t)
        input_size = t[:-1]
        output_size = t[-1]
        # [input_size, output_size] = m.get_shape().as_list()
        input_shape = tf.shape(x)
        batch_rank = input_shape.get_shape()[0].value - 1
        print(batch_rank)
        batch_shape = input_shape[:batch_rank]
        print(batch_shape)
        print(output_size)
        output_shape = tf.concat(0, [batch_shape, [output_size]])

        x = tf.reshape(x, [-1, input_size])
        y = tf.matmul(x, m)

        y = tf.reshape(y, output_shape)

        return y

    def batch_vm3(x, m):
        [input_size, output_size] = m.get_shape().as_list()

        input_shape = tf.shape(x)
        batch_rank = input_shape.get_shape()[0].value - 1
        batch_shape = input_shape[:batch_rank]
        # output_shape = tf.concat(0, [batch_shape, [output_size]])

        x = tf.reshape(x, [-1, input_size])
        y = tf.matmul(x, m)

        # y = tf.reshape(y, output_shape)
        y = tf.reshape(y, [2,4])
        return y

    a = np.arange(16, dtype=np.int32).reshape((2,4,2))
    b = np.arange(8, dtype=np.int32).reshape((2,4))

    r = batch_vm3(tf.Variable(a), tf.Variable(b))

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)
    result = sess.run(r)
    print(result)

if __name__ == "__main__":
    # test_multiply()
    # test_batch_matrix_vector()
    # test_get_shape_value()
    test_batchvm()