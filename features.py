# -*- coding: utf-8 -*-

import numpy as np
import math
import glob
import os
from numpy import genfromtxt
import pandas as pd


def cal_features(data):
    """
    :param: data: 训练数据
    :return: d: 栅格与发射机距离
              PL： Cost 231-Hata结果
              delta_h: 栅格与信号线相对高度
    """
    # cell_x: 站点所在栅格 X 坐标
    cell_x = data[:, 1]
    # cell_y: 站点所在栅格 Y 坐标
    cell_y = data[:, 2]
    # height_s: 站点发射机相对地面高度
    height_s = data[:, 3]
    # theta_a: 发射机水平方向角
    theta_a = data[:, 4]
    # theta_ed: 发射机垂直电下倾角
    theta_ed = data[:, 5]
    # theta_md: 发射机垂直机械下倾角
    theta_md = data[:, 6]
    # freq: 发射机中心频率
    freq = data[:, 7]
    # power: 发射机发射功率
    power = data[:, 8]
    # altitude_c: 站点所在海拔高度
    altitude_c = data[:, 9]
    # height_cell_b: 站点所在栅格的建筑物高度，没有则为0
    height_cell_b = data[:, 10]
    # clutter_c: 站点所在栅格地物类型索引[1, 2, ..., 20]
    clutter_c = data[:, 11]
    # obj_x: 目标栅格 X 坐标
    obj_x = data[:, 12]
    # obj_y: 目标栅格 Y 坐标
    obj_y = data[:, 13]
    # altitude_obj: 目标栅格海拔高度
    altitude_obj = data[:, 14]
    # height_obj_b: 目标栅格上建筑物高度，没有则为0
    height_obj_b = data[:, 15]
    # clutter_obj: 目标栅格地物类型索引[1, 2, ..., 20]
    clutter_obj = data[:, 16]
    # Cost 231-Hata 特征计算
    # alpha: 用户天线高度纠正项(待改)
    alpha = 0
    # 栅格与发射机距离d (栅格每个间隔固定为5m)
    d = 5 * np.sqrt(np.square(cell_x - obj_x) + np.square(cell_y - obj_y))

    # c_m: 场景纠正常数，根据场景赋值
    # 0 dB， 中等城市或郊区
    # 3 dB， 城市中心区
    c_m = np.zeros(len(clutter_obj))
    for index in np.where(clutter_obj == 20):
        c_m[index] = 3
    # 基站天线有效高度
    h_b = height_s + altitude_c - altitude_obj
    for index in np.where(h_b <= 0):
        h_b[index] = 1  # 本为0，log10（1）为0，故设为1
    # 用户天线有效高度
    h_ue = height_obj_b
    for index in np.where(h_ue <= 0):
        h_ue[index] = 1
    # frequency
    for index in np.where(freq <= 0):
        freq[index] = 1  # 本为0，log10（1）为0，故设为1
    # d
    for index in np.where(d <= 0):
        d[index] = 1  # 本为0，log10（1）为0，故设为1
    PL = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(h_b) - alpha + (44.9 - 6.55 * np.log10(h_ue)) * np.log10(d) + c_m

    # 栅格与信号线相对高度
    delta_h = height_s + altitude_c - altitude_obj - d * np.tan(theta_ed + theta_md)
    d = np.array(d, dtype="float32")
    PL = np.array(PL, dtype="float32")
    delta_h = np.array(delta_h, dtype="float32")

    return d, PL, delta_h


def cal_pcrr(y_true, y_pred):
    """
    计算 PCRR
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    t = -103
    tp = len(y_true[(y_true < t) & (y_pred < t)])
    fp = len(y_true[(y_true >= t) & (y_pred < t)])
    fn = len(y_true[(y_true < t) & (y_pred >= t)])
    if tp + fp == 0 or tp + fn == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    pcrr = 2 * (precision * recall) / (precision + recall)
    return pcrr


if __name__ == "__main__":
    os.chdir("./train_set")

    inst, label = [], []
    test_X, test_y = [], []

    for file in glob.glob("*.csv"):
        my_data = genfromtxt(file, delimiter=',')[1:]
        test_X = test_X + list(my_data[:100, :-1])
        test_y = test_y + list(my_data[:100, -1])
        break

    # test_X = np.array(test_X)
    # test_y = np.array(test_y)
    # for f in cal_features(test_X):
    #     test_X = np.column_stack((test_X, f))
    # print(test_X)

    y_pred = np.random.uniform(-100, -110, size=len(test_y))
    test_y = np.array(test_y)
    print(cal_pcrr(test_y, y_pred))
