#相比正向的神经网络 此处需要增设BP 反向传播
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat

path = "E:\\Apollo\\资料\\Coursera-ML-AndrewNg-Notes-master\\code\\ex4-NN back propagation\\ex4data1.mat"
data = loadmat(path)
X = data['X']
y = data['y']
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
print(y_onehot.shape)
#BP神经网络的组成： 1.输入为样本属性， 房屋面积 地理位置 建造时间 记为 X  2.通过权值得 Z(1) = θ^T * X  3. A(1) = sigmoid(Z(1)) 激活函数选取sigmoid
#                 4.输出的A(1)作为下一层的输入  5.计算A(2) 直到传到最后一层  6.反向传播 background algorithm  从最后一层的误差向前计算 链式求导
#                 7.代价函数 J(θ)
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
# input_size 输入个数， hidden_size 隐层数目， 隐层数目 * 每一层的输入数构成 theta
def Cost_Func(params, input_size, hidden_size, num_labels, X, y, learningRate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    #隐藏层数 * （输入的theta数 + 1）
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    J = 0
    for i in range(m):
        part1 = np.multiply(-y[i,:], np.log(h[i,:]))
        part2 = np.multiply(1 - y[i,:], np.log(1 - h[i,:]))
        J += np.sum(part1 - part2)

    J = J / m
    return J

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h

def differntial_g_z(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))
#params  input_size - 输入数目  hidden_size - 隐层数目  num——labels 输出的标签数
def BackGround(params, input_size, hidden_size, num_labels, X, y, learningRate):
    X = np.matrix(X)
    y = np.matrix(y)
    m = X.shape[0]


    theta1 = np.matrix(np.reshape(params[: (input_size + 1) * hidden_size], (hidden_size, input_size + 1)))         #theta1.T 为 400 + 1 * 25
    theta2 = np.matrix(np.reshape(params[(input_size + 1) * hidden_size :], (num_labels, hidden_size + 1)))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # 反向传播公式
    # 偏COST/偏theta2 = (A3 - y)*A3      其中小括号内为 偏cost偏z3
    # 偏cost/偏theta1 = [(A3 - y) * theta2 * g'(z2)] * A1  其中中括号内为 偏cost偏z2
    # 1.循环计算一下cost 2.循环计算一下theta1更新和theta2更新，然后concatenate二者
    J = 0
    i_theta1 = np.zeros(theta1.shape)
    i_theta2 = np.zeros(theta2.shape)
    for i in range(m):
        part1 = np.multiply(-y[i,:], np.log(h[i,:]))        # y 5000 * 10 & h 5000 * 10 为multiply
        part2 = np.multiply((1 - y[i, :]), np.log(1 - h[i,:]))
        J += np.sum(part1 - part2)
    # reg = (float(learningRate) / 2 * m) * (np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2)))
    # J = J / m + reg
    J += (float(learningRate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    #计算完cost 计算梯度反向传播
    for t in range(m):
        a1t = a1[t,:]   # 1 * 400 + 1    theta1.T = 401 * 25
        z2t = z2[t,:]   # 1 * 25
        a2t = a2[t,:]   # 1 * 26    sigmoid z2 补充偏置列
        # theta2.T = 26 * 10

        d3t_z = h[t,:] - y[t,:]   # 1 * 10
        z2t = np.insert(z2t, 0, values = np.ones(1)) #为了后续计算补充 1列从 1*25变1*26
        d2t_z = np.multiply(d3t_z*theta2, differntial_g_z(z2t)) # 1 * 26

        i_theta1 = i_theta1 + (d2t_z[:, 1:]).T * a1t # 25 * 1 * 1 * 401
        i_theta2 = i_theta2 + d3t_z.T * a2t # 10 * 26
    i_theta1 = i_theta1 / m
    i_theta2 = i_theta2 / m

    #正则化 i_theta + learningRate * theta / m
    i_theta1[:,1:] = i_theta1[:,1:] + (theta1[:,1:] * learningRate) / m
    i_theta2[:,1:] = i_theta2[:, 1:] + (theta2[:, 1:] * learningRate) / m

    res = np.concatenate((np.ravel(i_theta1), np.ravel(i_theta2)))
    return J, res


# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# 随机初始化完整网络参数大小的参数数组
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

# print(forward_propagate(X, theta1, theta2))
print(Cost_Func(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate))
m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)

# 将参数数组解开为每个层的参数矩阵

J, grad = BackGround(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print(J, grad.shape)

from scipy.optimize import minimize

# minimize the objective function
fmin = minimize(fun=BackGround, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})
print(fmin)

X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)
correct = [1 if a==b else 0 for (a,b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print("精度为： {0}%".format(accuracy * 100))