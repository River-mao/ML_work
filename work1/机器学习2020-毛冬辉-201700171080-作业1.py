import pandas as pd
import numpy as np
import random
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron(object):

    def __init__(self):
        self.max_iteration = 5000
        self.learning_step = 0.00001

    def train(self, features, labels):
        # train_features<--features
        # 定义对偶感知机更新参数
        N = len(labels)
        self.alpha = [0.0]*N
        self.b = 0
        self.Garma = [[0]*N]*N

        for i in range(N):
            for j in range(N):
                self.Garma[i][j] = sum(features[i]*features[j])
                if j%100 ==0:
                    precent = (i*N+j)/(N*N)
                    sys.stdout.write("\r Garma has been computed: %.2f%%" %(100*precent))
                    sys.stdout.flush()

        print("\r Garma has been computed")

        correct_count = 0
        time = 0

        while time < self.max_iteration:
            #随机抽取训练数据
            index = random.randint(0, N-1)
            #current_x = list(features[index])
            current_y = labels[index]
            fx = sum(self.alpha[j]*labels[j]*self.Garma[j][index] for j in range(N))+self.b
            #fx = sum([self.alpha[j]*labels[j]*sum(features[j]*current_x) for j in range(len(labels))])+ self.b

            if current_y*fx > 0:
                correct_count +=1
                sys.stdout.write("\r we want 5000 correct,we achieve %.2f%%" %(100*correct_count/self.max_iteration))
                sys.stdout.flush()

                if correct_count > self.max_iteration:
                    break
                continue
            else:
                self.alpha[index] += self.learning_step
                self.b += self.learning_step*current_y

        self.w = sum(self.alpha[i] * labels[i] * features[i] for i in range(len(labels)))

    def predict_(self,test_features):

        fx = sum([self.w[j]*test_features[j] for j in range(len(self.w))]) + self.b
        return int(fx > 0)

    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            labels.append(self.predict_(x))

        return labels

print('Start read data')

time_1 = time.time()

raw_data = pd.read_csv('./train_binary.csv', header=0)
# raw data shape =(42000, 785)
data = raw_data.values
#print("raw_data shape:",np.shape(raw_data))

# 像素点
# img shape = (42000, 784)
imgs = data[:10000, 1:]

#print("imgs shape:",np.shape(imgs))
# 标签
# labels shape = (42000, 1)
labels = data[:10000, 0]
#print("labels shape:",np.shape(labels))

# 选取 2/3 数据作为训练集， 1/3 数据作为测试集
train_features, test_features, train_labels, test_labels = train_test_split(
    imgs, labels, test_size=0.33, random_state=23323)

time_2 = time.time()
print('read data cost {}'.format(time_2 - time_1))

print('Start training')
p = Perceptron()
p.train(train_features, train_labels)

time_3 = time.time()
print("\r training cost {}".format(time_3 - time_2))

print('Start predicting')
test_predict = p.predict(test_features)
time_4 = time.time()
print('predicting cost {}'.format(time_4 - time_3))

score = accuracy_score(test_labels, test_predict)
print('The accruacy socre is {}'.format(score))

