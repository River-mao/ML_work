import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core import debugger
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

debug = debugger.Pdb().set_trace

#%matplotlib inline
# 鸢尾花(iris)数据集
# 数据集内包含 3 类共 150 条记录，每类各 50 个数据，
# 每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，
# 可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。
# 这里只取前100条记录，两项特征，两个类别。

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[0:60, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,1:2], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)


class BoostingTree(object):

    def __init__(self,X_train, y_train, max_iteration = 5000, min_error = 0.001):
        #数据个数
        self.N = int(len(y_train))
        #初始化权重矩阵，平均分配
        self.D = np.array(self.N * [float(1/self.N)])

        #当迭代次数大于最大迭代次数
        #或者分类误差小于最小可接受误差时
        #结束学习过程
        #最大迭代次数
        self.max_iteration = max_iteration
        #最小可接受误差
        self.min_error = min_error

        #alpha矩阵，每个弱分类器的系数
        #最多可有max_iteration个alpha_i
        self.alpha = np.zeros(self.max_iteration)

        #决策树桩矩阵
        #见cacl_v_list(X_train,v_number)函数初始化
        self.v_list = None
        #决策值的索引列表
        self.index_list = self.max_iteration*[0]

    def print_init(self):
        print("---------------------------------")
        print("-----------初始化----------------")
        print("---------------------------------")
        print("数据量:{}".format(len(y_train)+len(y_test)))
        print("训练数据量:{}".format(len(y_train)))
        print("测试数据量:{}".format(len(y_test)))
        print("D={}".format(self.D))

    def print_each_train(self,iteration,min_error_index,):
        print("-------------------------------------")
        print("-------------第{}个决策树-------------".format(iteration+1))
        print("-------------------------------------")
        print("决策树桩：v = {}".format(self.v_list[min_error_index]))
        print("权重：alpha = {}".format(self.alpha[iteration]))
        print("权值向量：D={}".format(self.D))

    def G_x(self, X_data, v):
        if X_data < v:
            return 1.0
        else:
            return -1.0

    def cacl_v_list(self,X_train,v_number=100):
        #self.v_list = X_train
        x_min = np.min(X_train)
        x_max = np.max(X_train)
        self.v_list = np.linspace(x_min, x_max,v_number)

    def update_D(self,alpha,X_train,y_train,index):
        Zm = 0.0
        for i in range(self.N):
            Zm += self.D[i]*np.exp(-alpha*y_train[i]*self.G_x(X_train[i],self.v_list[index]))
        for i in range(self.N):
            self.D[i] = self.D[i]*np.exp(-alpha*y_train[i]*self.G_x(X_train[i],self.v_list[index]))/Zm

    def CalcMinErrorAndInex(self,X_train,y_train):

        min_error = 1000
        min_error_index = 0

        # 对v_list的每个v值进行迭代
        # 求出最小的error与返回使error最小的v值在v_list中的index
        for i in range(len(self.v_list)):
            current_error = 0.00001
            for j in range(self.N):
                G_i_x = self.G_x(X_train[j], self.v_list[i])

                if G_i_x != y_train[j]:
                    current_error += self.D[j]

            if current_error < min_error:
                min_error = current_error
                min_error_index = i

        return min_error,min_error_index

    def train(self,X_train,y_train):
        self.print_init()
        #iteration的大小，决定分类树桩的个数
        for iteration in range(self.max_iteration):
            min_error, min_error_index = self.CalcMinErrorAndInex(X_train, y_train)

            #如果当前误差小于可接受最小误差，结束学习
            if (min_error <= self.min_error):
                break

            #记录本次迭代分类最小误差对应的v的索引
            self.index_list[iteration]= min_error_index
            self.alpha[iteration] = 0.5*np.log((1-min_error)/min_error)
            self.update_D(self.alpha[iteration],X_train,y_train,self.index_list[iteration])
            self.print_each_train(iteration,min_error_index)

    def predict(self,X_test,y_test):

        fx = 0
        right_num = 0
        for i in range(len(X_test)):
            for j in range(len(self.alpha)):
                fx += self.alpha[j]*self.G_x(X_test[i],self.v_list[self.index_list[j]])

            if np.sign(fx) == y_test[i]:
                right_num+=1
            fx = 0
        return right_num/len(y_test)


boostinftree = BoostingTree(X_train,y_train,max_iteration = 5, min_error = 0.001)
boostinftree.cacl_v_list(X_train,v_number=5)
boostinftree.train(X_train,y_train)
accuracy = boostinftree.predict(X_test,y_test)
print("测试正确率为：{}".format(accuracy))




