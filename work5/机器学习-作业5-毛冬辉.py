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
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:1], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=65)


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

    def G_x(self, X_data, v):
        if X_data>v:
            return 1
        else:
            return -1

    def cacl_v_list(self,X_train,v_number=100):
        x_min = np.min(X_train)
        x_max = np.max(X_train)
        self.v_list = np.linspace(x_min, x_max,v_number)

    def update_D(self,alpha,X_train,y_train,index):
        Zm = 0.0
        for i in range(self.N):
            Zm += self.D[i]*np.exp(float(alpha*y_train[i]*self.G_x(X_train[i],self.v_list[index])))
        for i in range(self.N):
            self.D[i] = self.D[i]*np.exp(float(alpha*y_train[i]*self.G_x(X_train[i],self.v_list[index])))/Zm

    def train(self,X_train,y_train):
        min_error = 1
        iteration = 0

        while(1):
            if (min_error <= self.min_error) | (iteration>=self.max_iteration):
                break

            index = 0
            current_error = 0.00001
            min_error_index = 0
            for i in range(len(self.v_list)):
                for j in range(self.N):
                    G_i_x = self.G_x(X_train[j],self.v_list[index])

                    if G_i_x != y_train[i]:
                        current_error += self.D[i]

                if current_error<min_error:
                    min_error = current_error
                    min_error_index = index
                index +=1

            #记录本次迭代分类最小误差对应的v的索引
            self.index_list[iteration]=min_error_index
            self.alpha[iteration] = 0.5*np.log((1-min_error)/min_error)
            self.update_D(self.alpha[iteration],X_train,y_train,self.index_list[min_error_index])

            iteration+=1



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


boostinftree = BoostingTree(X_train,y_train,max_iteration = 10, min_error = 0.001)
boostinftree.cacl_v_list(X_train,v_number=50)
boostinftree.train(X_train,y_train)
accuracy = boostinftree.predict(X_test,y_test)
print("the accuracy is {}".format(accuracy))




