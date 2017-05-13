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
@file: testGather.py
@time: 17-5-13 上午8:21
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


def  test_variable(_variable):
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    v = sess.run(_variable)
    print(v)
    print(v.shape)


def test_gathernd():
    '''
    tf.gather_nd()函数使用indice参数的前N-1维与之前的tf.gather的意义相同,第N维上的内容即为在param数组中
    的索引,如果是一个元素则由N-1维的tensor的该位置将填充param[",".join(第N维的内容)]
    :return:
    '''
    a = np.arange(12).reshape((2,2,3))
    print(a)

    indice0 = [[[1],[0]]]
    indice1 = [[[0,0], [1,1]]]
    indice2 = [[[0,0,0], [1,1,2]]]

    v0 = tf.gather_nd(a, indice0)
    v1 = tf.gather_nd(a, indice1)
    v2 = tf.gather_nd(a, indice2)
    test_variable(v0)
    test_variable(v1)
    test_variable(v2)


def test_scatternd():
    '''
    indice类是gather_nd,如果对于要更新的tensor有多个update元素与其对应,则选择最新的更新
    :return:
    '''
    a = np.zeros((2,2,3))
    f = tf.Variable(a)
    print(a)

    indice0 = [[[1],[0]], [[1],[0]]]
    updates = [[[1,2,3],[4,5,6]], [[7,8,9],[1,1,1]]], [[[1,2,3],[0,5,6]], [[7,8,9],[2,2,2]]]

    indice0 = [[[1]], [[0]]]
    updates = [[[1, 2, 3], [4, 5, 6]]], [ [[7, 8, 9], [2, 2, 2]]]
    u = tf.scatter_nd_update(f, indice0, updates)
    test_variable(u)


if __name__ == "__main__":
    test_scatternd()