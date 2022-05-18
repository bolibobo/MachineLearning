#正则化 ： 多项式线性回归的常用评价方法，为了避免过拟合，使得多项式各项参数之和
#sum(theta)作为代价函数的一项，由于cost func需要最小，因此可以限制过拟合程度
#线性回归的多项式一般是样本!!!属性少!!!时有效，否则考虑神经网络
#g(θ) = θ0 + θ1*x1 + θ2*x2 + θ3*x1^2*x2 + θ4*x1*x2^2 + θ5*... + θn*x1^(t)*x2^(n-t)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import linear_model

def sigmoid(X, theta):
    return 1 / (1 + np.exp(-X * theta.T))

def update_gradient_regularization(theta, X, y, mlambda):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)

    parameters = theta.shape[1]
    tmp = np.zeros((parameters, 1))
    errors = sigmoid(X, theta) - y

    for i in range(parameters):
        term = np.multiply(errors, X[:, i])
        if (i == 0):
            tmp[i] = np.sum(term) / len(X)
        else:
            tmp[i] = (np.sum(term) + mlambda * theta[:,i]) / len(X)
    return tmp

def cost_func(theta, X, y, mlambda):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)

    part1 = np.multiply(-y, np.log(sigmoid(X, theta)))
    part2 = np.multiply((1 - y), np.log(1 - sigmoid(X, theta)))
    reg = mlambda * np.sum(np.power(theta[:, 1:theta.shape[1]], 2)) / (2 * len(X))
    return np.sum(part1 - part2) / len(X) + reg

def binary_condition(theta, X):
    h_x = sigmoid(X, theta)
    h_x = np.matrix(h_x)
    return (1 if x >= 0.5 else 0 for x in h_x)

path = "E:\\Apollo\\资料\\Coursera-ML-AndrewNg-Notes-master\\code\\ex2-logistic regression\\ex2data2.txt"
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admmitted'])
positive = data[data['Admmitted'].isin([1])]
negative = data[data['Admmitted'].isin([0])]
# data.insert(0, 'Ones', 1)

#构造特征多项式 x^10 + ... + x^0 (1)
degree = 5
x1 = data['Exam1']
x2 = data['Exam2']

for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)


data.drop('Exam1', axis=1, inplace=True)
data.drop('Exam2', axis=1, inplace=True)
data.insert(1, 'Ones', 1) #用于和 theta0相乘，得到线性回归常数项
# print(data.head())


cols = data.shape[1]
X = data.iloc[:, 1:cols]
y = data.iloc[:, 0:1]
X = X.values
y = y.values
mlambda = 1
theta = np.zeros((1, 11))
# print(X.shape)

# print(cost_func(theta, X, y, mlambda))
# print(update_gradient_regularization(theta, X, y, mlambda))
res_theta = opt.fmin_tnc(func=cost_func, fprime=update_gradient_regularization, x0=theta, args=[X,y,mlambda])
# print(res_theta)
res_y = binary_condition(np.matrix(res_theta[0]), X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a,b) in zip(res_y, y)]
correct = np.matrix(correct)
sum1 = np.sum(correct)
print("accuracy = {0}%".format(sum1 * 100 / len(y)))

model = linear_model.LogisticRegression(penalty='l2',C=1.0)  #正则化L2 平方和 c就是权重
model.fit(X, y.ravel())
print("accuracy by Function Library is {0}%".format(model.score(X, y)*100))


#fmin_tnc使用截断牛顿法，是opt.minimize的特殊形式，可显示迭代步骤  后者为通用求解器，性能更好，但不显示迭代步骤