import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

path = "E:\\Apollo\\资料\\Coursera-ML-AndrewNg-Notes-master\\code\\ex2-logistic regression\\ex2data1.txt"
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admmitted'])
positive = data[data['Admmitted'].isin([1])]
negative = data[data['Admmitted'].isin([0])]
# fig,ax = plt.subplots(figsize = (12,8))
# ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b',marker='o',label = 'Admmitted')
# ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r',marker='*',label = 'NOT Admmitted')
# ax.legend()
# ax.set_xlabel('Exam1')
# ax.set_ylabel('Exam2')
# plt.title('Initial Data', x = 0.5, y = 1)
# plt.show()

theta = np.zeros((1, 3))
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]
X = X.values
y = y.values

def sigmoid(h_fun_value):
    return 1 / (1 + np.exp(-h_fun_value))

def h_fun(Sample, theta):               # (m,1)数据，逻辑回归使用sigmoid(h(theta))
    Sample = np.matrix(Sample)
    theta = np.matrix(theta)
    paramters = len(X)
    res = np.zeros((paramters, 1))
    for i in range(paramters):
        res[i] = sigmoid(Sample[i,:] * theta.T)
    return res
# print(h_fun(X,theta))
# print(h_fun(X,theta).shape)
def cost_fun(theta, X, y):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)

    part1 = np.multiply(-y, np.log(h_fun(X, theta)))
    part2 = np.multiply((1-y), np.log(1 - h_fun(X, theta)))
    return np.sum((part1 - part2)) / len(X)

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = theta.shape[1]
    error = h_fun(X, theta)-y

    tmp = np.zeros(parameters)
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        tmp[i] = np.sum(term) / len(X)
    return tmp

def binary_condition(theta, X):
    h_x = h_fun(X, theta)
    h_x = np.matrix(h_x)
    return (1 if x >= 0.5 else 0 for x in h_x)

res_theta = opt.fmin_tnc(func=cost_fun, x0=theta, fprime=gradient, args=(X,y))
print(cost_fun(theta,X,y))
print(cost_fun(res_theta[0],X,y))

res_y = binary_condition(np.matrix(res_theta[0]), X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a,b) in zip(res_y,y)]
c1 = np.matrix(correct)
print("accuracy = {0}%".format(np.sum(c1)))
# for i in range(iters):
#     theta = theta - alpha * gradient(X, y, theta)
#     print(cost_fun(X, y, theta))
