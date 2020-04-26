# encoding=utf-8

import cv2
import time
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

total_class = 10


def log(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.debug('start %s()' % func.__name__)
        ret = func(*args, **kwargs)

        end_time = time.time()
        logging.debug('end %s(), cost %s seconds' % (func.__name__, end_time - start_time))

        return ret

    return wrapper


# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img


@log
def binaryzation_features(trainset):
    features = []

    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        img_b = binaryzation(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(img_b)

    features = np.array(features)
    features = np.reshape(features, (-1, 784))

    return features


class Tree(object):
    def __init__(self, node_type, Class=None, feature=None):
        self.node_type = node_type
        self.dict = {}
        self.Class = Class
        self.feature = feature

    def add_tree(self, val, tree):
        self.dict[val] = tree

    def predict(self, features):
        if self.node_type == 'leaf':
            return self.Class

        tree = self.dict[features[self.feature]]
        return tree.predict(features)


def calc_ent(x):
    """
        calculate shanno ent of x
        计算经验熵
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        # x[x == x_value].shape[0] 计算x中元素值为x_value的个数
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def calc_condition_ent(x, y):
    """
        calculate ent H(y|x)
        计算条件经验熵
    """
    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent


def calc_ent_grap(x, y):
    """
        calculate ent grap
    """
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap
#-------------------------------------------------------------------------------
#---------------------将ID3改为C45的关键部分--------------------------------------
#---------------------计算数据集D关于特征A的熵------------------------------------
def calc_ent_had(A):
    '''
    calculate entropy for D with feature:A
    '''
    A = np.array(A)
    feature_value_list = set([A[i] for i in range(A.shape[0])])

    # 由于 ent_hda 在计算信息增益比的时候在分母位置，将其赋值为一个非0值，避免出发错误
    ent_hda = 0.0000001
    for feature_value in feature_value_list:
        p = float(A[A == feature_value].shape[0])/float(A.shape[0])
        logp = np.log2(p)
        ent_hda -= p * logp

    return ent_hda

def recurse_train(train_set, train_label, features, epsilon):
    global total_class

    LEAF = 'leaf'
    INTERNAL = 'internal'

    # 步骤1——如果train_set中的所有实例都属于同一类Ck
    label_set = set(train_label)  # set返回list中无序不重复的数
    if len(label_set) == 1:
        return Tree(LEAF, Class=label_set.pop())

    # 步骤2——如果features为空
    (max_class, max_len) = max([(i, len(list(filter(lambda x: x == i, train_label)))) for i in range(total_class)],
                               key=lambda x: x[1])

    if len(list(features)) == 0:
        return Tree(LEAF, Class=max_class)

    # 步骤3——计算信息增益比
    max_feature = 0
    max_grda = 0

    D = train_label
    HD = calc_ent(D)

    for feature in features:
        A = np.array(train_set[:, feature].flat)
        gda = HD - calc_condition_ent(A, D)

        #----------------------------------------------------------
        #----------------计算训练数据集D关于特征A的值的熵HA(D)--------
        ent_hda = calc_ent_had(A)
        # ---------------计算信息增益比-----------------------------
        grda = gda/ent_hda

        if grda > max_grda:
            max_grda, max_feature = grda, feature

    # 步骤4——小于阈值
    if max_grda < epsilon:
        return Tree(LEAF, Class=max_class)

    # 步骤5——构建非空子集
    sub_features = list(filter(lambda x: x != max_feature, features))  # 去掉当前特征
    tree = Tree(INTERNAL, feature=max_feature)

    feature_col = np.array(train_set[:, max_feature].flat)
    feature_value_list = set([feature_col[i] for i in range(feature_col.shape[0])])
    for feature_value in feature_value_list:

        index = []
        for i in range(len(train_label)):
            if train_set[i][max_feature] == feature_value:
                index.append(i)

        sub_train_set = train_set[index]
        sub_train_label = train_label[index]

        sub_tree = recurse_train(sub_train_set, sub_train_label, sub_features, epsilon)
        tree.add_tree(feature_value, sub_tree)

    return tree


@log
def train(train_set, train_label, features, epsilon):
    return recurse_train(train_set, train_label, features, epsilon)


@log
def predict(test_set, tree):
    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # https://github.com/WenDesi/lihang_book_algorithm/blob/master/data/test.csv
    raw_data = pd.read_csv('./train.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 图片二值化
    features = binaryzation_features(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
                                                                                random_state=23323)

    tree = train(train_features, train_labels, [i for i in range(784)], 0.1)
    test_predict = predict(test_features, tree)
    score = accuracy_score(test_labels, test_predict)

print("The accruacy socre is {}".format(score))


