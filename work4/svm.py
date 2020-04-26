import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core import debugger
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

debug = debugger.Pdb().set_trace

#%matplotlib inline
# 问题描述：实现SVM的SMO优化算法，训练给定数据集，并测试和可视化结果。对比与sklearn科学计算工具包的svm方法效果。


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
    return data[:,:2], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=65)


class SVM(object):

    def __init__(self, kernel='linear', epsilon=0.001, C=1.0, Max_Interation=5000):
        self.kernel = kernel
        self.epsilon = epsilon
        self.C = C
        self.Max_Interation = Max_Interation

    def _init_parameters(self, features, labels):
        # 参数初始化
        self.X = features
        self.Y = labels

        self.b = 0.0
        self.n = len(features[0])
        self.N = len(features)
        self.alpha = [0.0] * self.N
        self.E = [self._E_(i) for i in range(self.N)]

    def _satisfy_KKT(self, i):
        ygx = self.Y[i] * self._g_(i)
        if abs(self.alpha[i]) < self.epsilon:
            return ygx >= 1
        elif abs(self.alpha[i] - self.C) < self.epsilon:
            return ygx <= 1
        else:
            return abs(ygx - 1) < self.epsilon

    def is_stop(self):
        for i in range(self.N):
            satisfy = self._satisfy_KKT(i)
            if not satisfy:
                return False
        return True

    def _select_two_parameters(self):
        # 选择两个变量
        index_list = [i for i in range(self.N)]

        i1_list_1 = list(filter(lambda i: self.alpha[i] > 0 and self.alpha[i] < self.C, index_list))
        i1_list_2 = list(set(index_list) - set(i1_list_1))

        i1_list = i1_list_1
        i1_list.extend(i1_list_2)

        for i in i1_list:
            if self._satisfy_KKT(i):
                continue

            E1 = self.E[i]
            max_ = (0, 0)

            for j in index_list:
                if i == j:
                    continue

                E2 = self.E[j]
                if abs(E1 - E2) > max_[0]:
                    max_ = (abs(E1 - E2), j)
            return i, max_[1]

    def _K_(self, x1, x2):
        # 核函数
        if self.kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        if self.kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1) ** 3
        print('没有定义核函数')
        return 0

    def _g_(self, i):
        # g(X[i])
        result = self.b

        for j in range(self.N):
            result += self.alpha[j] * self.Y[j] * self._K_(self.X[i], self.X[j])
        return result

    def _E_(self, i):
        # E(i)
        return self._g_(i) - self.Y[i]

    def clip_LH(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    def thresh_LH(self, i1, i2):
        if self.Y[i1] == self.Y[i2]:
            L = max(0, self.alpha[i2] + self.alpha[i1] - self.C)
            H = min(self.C, self.alpha[i2] + self.alpha[i1])
        else:
            L = max(0, self.alpha[i2] - self.alpha[i1])
            H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])
        return L, H

    # 作业一：完成下面的训练过程，补充缺失代码。用???标注。
    def train(self, features, labels):

        self._init_parameters(features, labels)

        for times in range(self.Max_Interation):

            # 启发式的选取两个样本index。
            i1, i2 = self._select_two_parameters()

            #a2_new_unc
            K11 = self._K_(self.X[i1],self.X[i1])
            K22 = self._K_(self.X[i2],self.X[i2])
            K12 = self._K_(self.X[i1],self.X[i2])
            K21 = self._K_(self.X[i2],self.X[i1])
            E1 = self.E[i1]
            E2 = self.E[i2]
            y1 = self.Y[i1]
            y2 = self.Y[i2]
            yita = K11 + K22 - 2 * K12
            alpha1_old = self.alpha[i1]
            alpha2_old = self.alpha[i2]

            #计算alpha2_new_unc
            alpha2_new_unc =alpha2_old + y2*(E1-E2)/yita
            #剪切到合法区间
            L, H  = self.thresh_LH(i1,i2)
            alpha2_new = self.clip_LH(alpha2_new_unc,L,H)

            #据a1y1+a2y2=sigma，得到a1_new。
            alpha1_new = alpha1_old + y1*y2*(alpha2_old-alpha2_new)

            # ??? 根据a1计算的b1
            b1_new = -E1-y1*K11*(alpha1_new-alpha1_old) - y2*K21*(alpha2_new-alpha2_old)+self.b
            # ??? 根据a2计算的b1
            b2_new = -E2-y1*K12*(alpha1_new-alpha1_old) - y2*K22*(alpha2_new-alpha2_old)+self.b

            if alpha1_new > 0 and alpha1_new < self.C:
                b_new = b1_new
            elif alpha2_new > 0 and alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            # 更新a、b和E
            self.alpha[i1] = float(alpha1_new)
            self.alpha[i2] = float(alpha2_new)
            self.b = b_new

            self.E[i1] = self._E_(i1)
            self.E[i2] = self._E_(i2)
        # print([int(i) for i in self.alpha])
        return 'train done!'

    def _predict_(self, feature):
        result = self.b

        for i in range(self.N):
            result += self.alpha[i] * self.Y[i] * self._K_(feature, self.X[i])

        if result > 0:
            return 1
        return -1

    def predict(self, features):
        results = []

        for feature in features:
            results.append(self._predict_(feature))

        return results

svm = SVM()
svm.train(X_train, y_train)
test_predict = svm.predict(X_test)
score = accuracy_score(y_test, test_predict)
print("the test score is:",score)  # 测试集准确率

#-----------------------------------------------------分割线----------------------------------------------

# 作业二：可视化以上优化好的分类超平面和间隔界面，最好把支持向量也标注来。

# matplotlib画图中中文显示会有问题，需要这两行设置默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

weight = [0.0,0.0]
for i in range(len(y_train)):
    weight += svm.alpha[i]*y_train[i]*X_train[i]
support_vector_list = []
for i in range(len(y_train)):
    if y_train[i]*(np.dot(weight,X_train[i])+svm.b)<=1:
        support_vector_list.append(X_train[i])

support_vector_list = np.array(support_vector_list)

plt.xlabel('X0')
plt.ylabel('X1')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train ,marker='o')
plt.scatter(X_test[:,0], X_test[:,1],c=y_test,marker='x')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
plt.xlim(xlim)
plt.ylim(ylim)
# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
Hyperplane_x1 = -(weight[0]/weight[1])*xx-svm.b/weight[1]
marginplane_1 = (1-weight[0]*xx-svm.b)/weight[1]
marginplane_2 = (-1-weight[0]*xx-svm.b)/weight[1]
plt.plot(xx,Hyperplane_x1,c = 'k',linestyle = '-')
plt.plot(xx,marginplane_1,c = 'b',linestyle = '--')
plt.plot(xx,marginplane_2,c = 'r',linestyle = '--')
# plot decision boundary and margins
# plot support vectors
ax.scatter(support_vector_list[:, 0], support_vector_list[:, 1], s=150,
           linewidth=1, facecolors='none', edgecolors='k')

plt.title("SMO算法")
plt.legend()
plt.show()

#-------------------------------------------------分割线-----------------------------------
#-----------------------------------------------------------------------------------------
# 作业三：用sklearn科学计算工具包   实现以上数据集的分类，并可视化分类超平面，间隔界面，最好把支持向量也标注来。from sklearn.svm import SVC
print(__doc__)
from sklearn.svm import SVC
#from sklearn.datasets import make_blobs
clf = SVC(C=1.0,kernel='linear',tol=1e-3,max_iter=5000)
clf.fit(X_train,y_train)
y_hat = clf.predict(X_test)
acc = accuracy_score(y_test,y_hat)
print("the test score is:",acc)  # 测试集准确率

"""
=========================================
SVM: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a Support Vector Machine classifier with
linear kernel.
"""

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o')

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='b', levels=[-1, 0, 1], alpha=1,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150,
           linewidth=1, facecolors='none', edgecolors='k')
ax.scatter(X_test[:,0],X_test[:,1],c=y_test, marker = 'x')
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('使用skearn库函数')
plt.show()
