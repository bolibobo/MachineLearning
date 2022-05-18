import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = "E:\\Apollo\\资料\\Coursera-ML-AndrewNg-Notes-master\\code\\ex1-linear regression\\ex1data1.txt"
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
#print(data.head())                 ##查看数据格式 前五行
#print(data.describe())             ##查看数据处理结果 求和 中间数 最小值等
#data.plot(title='Input Data', kind='scatter', x='Population', y='Profit', figsize=(12,8))
#plt.show()

#代价函数（基于线性拟合的均方误差）  J = 1 / 2m * sum[(h(xi) - yi)^2]
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

#梯度下降法 h(x) = θ1 * x1 + .. + θk * xk + θ0  对θj求偏导为xj
# θj := θj - α*偏导J(θj)  ---  θj：= θj - α * 1/m*error*xj
def gradientDescent(X, y, theta, alpha, iters):     #alpha步长 iters迭代次数
    temp = np.matrix(np.zeros(theta.shape))
    #print("temp = ", temp)
    parameters = theta.shape[1]                 #循环次数（等于线性拟合的参数theta个数）
    cost = np.zeros(iters)                      #记录每次的cost值

    #循环批量进行梯度下降
    for i in range(iters):
        error = (X * theta.T) - y                   #theta是1 * 2
        #error是代价函数J(θ)值   m * n * n * 1 = m * 1

        #θj：= θj - 1/m*J(θ)*x  term = J(θ) * xj
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            print(term.shape)
            temp[0, j] =  theta[0, j] - np.sum(term) * alpha / len(X)
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost


data.insert(0, 'Ones', 1)       ##加入一列全为1，列名为 Ones
#print(data.head())
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]
#print(data.shape[1])

X = np.matrix(X.values)         ##转数据为矩阵
tmp = X[:,1:X.shape[1]]
#print(tmp)
y = np.matrix(y.values)
#theta = np.matrix(np.array([0,0]))  ##一行两列

theta = np.zeros((1,2))
alpha = 0.01
iters = 1        ##步长和迭代次数
g, cost = gradientDescent(X, y, theta, alpha, iters)


print("初始状态时均方误差: ",computeCost(X, y, theta))
print("训练结束时均方误差: ",computeCost(X, y, g))
print("参数为： ", g)

#绘图
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0,0] + (g[0,1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

#print(data.head())
#print(X)