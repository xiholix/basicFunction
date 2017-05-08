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
@file: segment.py
@time: 17-5-8 下午8:44
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


def test_segment_sum():
    a = np.arange(16).reshape((4,4))
    # b = np.array([0,1,0,1])
    '''
    segment ids are not increasing, 如果ids不是增序就用unsorted_segment_sum
    '''
    b = np.array([0,0,1,1])

    result = tf.segment_sum(a, b)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    r = sess.run(result)
    print(r)


def test_unsorted_segment_sum():
    a = np.arange(16).reshape((4,4))
    b = np.array([0,1,0,1])

    result = tf.unsorted_segment_sum(a, b, 4)
    #好像后面的第三个参数并没有什么作用

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    r = sess.run(result)
    print(r)

if __name__ == "__main__":
    test_segment_sum()