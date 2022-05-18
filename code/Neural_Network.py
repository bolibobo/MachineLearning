import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn import linear_model

data = loadmat("E:\\Apollo\\资料\\Coursera-ML-AndrewNg-Notes-master\\code\\ex3-neural network\\ex3data1.mat")
# print(data)
# print(data['X'].shape)
# print(data['y'].shape)

def sigmoid(z):                                # 1 / (1 + e^-h(θ))
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, learningRate):           # J(θ) = 1/m*np.sum(-ylog(h(Z) - (1-y)log(1-h(Z))) + reg
    theta = np.matrix(theta)                   #其中 Z = θi^T*X+b 正则化 reg = np.sum(θi ^ 2) / 2*m
    X = np.matrix(X)
    y =np.matrix(y)

    part1 = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    part2 = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = learningRate * np.sum(np.power(theta[:, 1:theta.shape[1]], 2)) / (2 * len(X))
    return np.sum(part1 - part2) / len(X) + reg
#梯度下降法                                     #寻找代价函数最小结果
def gradient_with_loop(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = theta.shape[1]
    tmp = np.zeros((parameters, 1))

    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if(i == 0) :
            tmp[i] = np.sum(term) / len(X)
        else :
            tmp[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])
    return tmp
#优化的梯度函数，和上面的梯度函数异曲同工，但是更适合BP神经网络 -- 数值求解反向传播公式得到的
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = theta.shape[1]
    #反向传播 第i层误差  theta * X - y 或者 X * theta.T - y   二者结果一样
    error = sigmoid(X * theta.T) - y
    #计算每层的梯度 1. 即 偏导J(θ)/偏导θ(i) -- 对第i层的θ求偏导
    #链式求导 偏J/偏A * 偏A/偏Z * 偏Z/偏θ   error简写为: A = H(Z) = sigmoid(z) 偏A/偏Z = A(1-A) 此处不证，sigmoid的性质
    #grad1 = (-y/A + (1-y)/(1-A)) * (A * (1-A)) * 偏Z/偏θ = (A - y) * 偏Z/偏θ
    #grad1 = (A - y) * 偏Z/偏θ = X.T * error
    #同理易得计算每层梯度第二部分 2.对b求偏导 结果为 λ * θi / m
    #grad = grad1 + grad2 = X.T * error + (λ * θi) / m
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    #且theta0的计算中不含正则项因此单列 重新赋值
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
    #print(grad.shape)  #看下类型
    return np.array(grad).ravel()  ###???为什么要转array，matrix可以直接转ravel吧

#多分类任务，数字一共10种，由于分类的输入类别为 1 - 10因此在后面判断是哪一类时要从1开始
def one_vs_all(X, y, num_labels, learningRate):     #num_labels分类器个数，对于全分类就是类别数10
    rows = X.shape[0]
    params = X.shape[1]

    # 10 * （样本属性数 + 1） = theta矩阵格式
    init_theta = np.zeros((num_labels, params + 1)) #params属性数目（b为常数偏移），即params + 1为每组分类器的theta数

    #插入样本 arr为插入目标，obj为位置，values为插入的数据，axis为期望的维度
    X = np.insert(arr = X, obj = 0, values=np.ones(rows), axis=1)

    for i in range(1, num_labels + 1):         #针对 每个分类器进行计算
        theta = np.zeros(params + 1)

        #此处为筛选样本属于哪一类 （标签从1开始 ？暂不明白）
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))       #转为 m * 1向量和计算的样本对应

        #利用opt计算
        fmin = opt.minimize(fun = cost, x0 = theta, args = (X, y_i, learningRate), method = 'TNC', jac = gradient)
        # print(fmin)
        init_theta[i - 1, :] = fmin.x
    # fmin_tnc使用截断牛顿法，是opt.minimize的特殊形式，可显示迭代步骤  后者为通用求解器，性能更好，但不显示迭代步骤
    return init_theta

rows = data['X'].shape[0]
params = data['X'].shape[1]

all_theta = np.zeros((10, params + 1))

X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

theta = np.zeros(params + 1)

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))
print(X.shape, y_0.shape, theta.shape, all_theta.shape)

np.unique(data['y'])
print(np.unique(data['y']))

all_theta = one_vs_all(data['X'], data['y'], 10, 1)
print(all_theta)


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax

y_pred = predict_all(data['X'], all_theta)
correct = [1 if a==b else 0 for (a,b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct))) / float(len(correct))
print('accuracy = {0}%'.format(accuracy * 100))