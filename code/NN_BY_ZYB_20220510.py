import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.io import loadmat


#1.sigmoid 函数
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
#2.梯度下降（循环）
def gradient_with_loop(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = theta.shape[1]
    tmp = np.zeros(parameters + 1)

    error = sigmoid(X * theta.T) - y
    for i in range(parameters + 1):
        term = np.multiply(X[:,i], error)
        if(i == 0):        #无reg
            tmp[i] = np.sum(term) / len(X)
        else:
            tmp[i] = np.sum(term) / len(X) + learningRate * np.sum(theta[:,i]) / len(X)
    return tmp
#3.梯度下降（批量）
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    params = int(theta.ravel().shape[1])      #二维的theta矩阵 转为向量求得所有参数值
    error = sigmoid(X * theta.T) - y         # m * 1              m * n
    grad = ((X.T * error) / len(X)).T + learningRate * theta / len(X)

    grad[0, 0] = np.sum(np.multiply(X[:,0], error)) / len(X)
    return grad

#4.损失函数 cost
def cost(theta, X, y, learningRate): # J(θ) = 1/m*np.sum(-ylog(h(Z) - (1-y)log(1-h(Z))) + reg
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    #  X * theta.t = 常数（计算结果）   part1 =  常数 .* 结果   .* 即 multiply
    part1 = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    part2 = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = learningRate * np.sum(np.power(theta[:, 1 : theta.shape[1]], 2)) / (2 * len(X))
    return np.sum(part1 - part2) / len(X) + reg
#5.优化  opt.minimize()
def one_vs_all(X, y, num_labels, learningRate):
    rows = X.shape[0]                               # m 测试集样本数
    params = X.shape[1]                             # n 样本属性数

    all_theta = np.zeros((num_labels, params + 1))  #一共 10 * 属性数目 + 1

    #插入常数项列
    X = np.insert(X, 0, values = np.ones(rows), axis = 1)
    for i in range(1, num_labels + 1):              # 样本的编号为 1 到 10
        theta_tmp = np.zeros(params + 1)
        #构建样本
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        fmin = opt.minimize(fun=cost, x0=theta_tmp, args=(X, y_i, learningRate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x
    return all_theta

#6.预测predict_all theta

def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    predict_res = np.zeros((rows, 1))
    #筛选出所有分类器里面值最大的，作为分类结果
    #all_theta = 10 * (n + 1)       X = m * (n + 1)
    calcu_res = sigmoid(X * all_theta.T)         # m * 10
    for i in range(rows):
        predict_res[i] = np.argmax(calcu_res[i,:], 1) + 1
    return predict_res

# rows = data['X'].shape[0]
# params = data['X'].shape[1]
# all_theta = np.zeros((10, params + 1))
# X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
# theta = np.zeros(params + 1)
# path = "E:\\Apollo\\资料\\Coursera-ML-AndrewNg-Notes-master\\code\\ex3-neural network\\ex3data1.mat"
# data = loadmat(path)

#测试BP和非BP的区别
path = "E:\\Apollo\\资料\\Coursera-ML-AndrewNg-Notes-master\\code\\ex4-NN back propagation\\ex4data1.mat"
data = loadmat(path)
X = data['X']
y = data['y']
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
# print(y_onehot.shape)


all_theta = one_vs_all(data['X'], data['y'], 25, 1)
print(all_theta)
y_pred = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))