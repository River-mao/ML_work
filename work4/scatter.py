# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
def scatter(X,y):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=9, xmin=0)
    plt.ylim(ymax=9, ymin=0)
    # 画两条（0-9）的坐标轴并设置轴标签x，y

    x1 = np.random.normal(2, 1.2, 300)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
    y1 = np.random.normal(2, 1.2, 300)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标
    x2 = np.random.normal(7.5, 1.2, 300)
    y2 = np.random.normal(7.5, 1.2, 300)
    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#DC143C'
    area = np.pi * 4 ** 2  # 点面积
    # 画散点图
    plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='类别A')
    plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='类别B')
    plt.plot([0, 9.5], [9.5, 0], linewidth='0.5', color='#000000')
    plt.legend()
#plt.savefig('work4\svm.png', dpi=300)
    plt.show()




""" 用于绘制超平面的函数
思想:生成网格点,并令分类器进行预测,最终将预测点和网格点合成绘制热力图,间接地表示出不同的区域
 """


def plot_hyperplane(clf, X, y,h=0.02,draw_sv=True,title='skearn'):
    # create a mesh to plot in
    X = np.mat(X)

    # 使用numpy的库函数找到最值,确定绘图范围
    x0_min, x0_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    x1_min, x1_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h),np.arange(x1_min, x1_max, h))

    plt.title(title)
    plt.xlim(x0.min(), x0.max())
    plt.ylim(x1.min(), x1.max())
    plt.xticks(())
    plt.yticks(())

    # x0,x1扁平化,生成二维坐标并将预测值作为函数值
    # SVM的分割超平面
    Z = clf.predict(np.c_[x0.ravel(), x1.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(x0.shape)
    # 绘制康拓图
    plt.contourf(x0, x1, Z, alpha=0.5)


    colors = ['#00CED1','#DC143C']
    # 去重
    labels = np.unique(y)


    # 对于每一个标签种类
    for label in labels:
        """ 推荐学习:Numpy数组切片、bool索引、掩码和花哨索引 """
        # 这是掩码索引
        if label == -1:
            index = 0
        else:
            index = 1
        plt.scatter(X[y == label][:, 0].tolist(),X[y == label][:, 1].tolist(),
                    c=colors[index],marker='o')
    # 画出支持向量
    if draw_sv:
        sv = clf.support_vectors_
        #print(sv)
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')


plot_hyperplane(svc, X, y)
plt.show()