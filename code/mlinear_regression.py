#线性回归 我这边用  h(x) = θ0 + θ1 * X1 + θ2 * X2+ ... +θk * Xk  xk为样本属性
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd            #操作数据csv的库

path = "E:\\Apollo\\资料\\Coursera-ML-AndrewNg-Notes-master\\code\\ex1-linear regression\\ex1data2.txt"
data = pd.read_csv(path, header = None, names=['Size', 'Bedrooms', 'Price'])
#print(data.head())  #检验下读到没有

data2 = (data - data.mean()) / data.std()
#print(data2.head())
#归一化      数据 - 均值 / 标准差      均值方差归一化，适合无明显边界数据


#定义两个函数 代价函数和梯度下降   m*n*n*1
#代价函数 计算公式   J = 1 /(2*m) * sum[(h(xi) - yi)^ 2]
def costCalculation(X, y, theta):
    tmp = np.power(X * theta[:,0:(theta.shape[1]-1)].T + theta[:,theta.shape[1]-1] - y, 2)
    return np.sum(tmp) / (2 * len(X))
#梯度下降 θj：= θj - α * 1/m* sum(error*xj)
def gradientDescent(X, y, theta, alpha, iters):
    tmp = np.zeros(theta.shape)    #存一下每次更新的theta
    paratemers = theta.shape[1]
    cost = np.zeros(iters)
    #mOne = np.ones((len(X),1));
    for i in range(iters):
        error = X * theta[:,0:(theta.shape[1]-1)].T + theta[:,theta.shape[1]-1] - y
        for j in range(paratemers - 1):
            term = np.multiply(error, X[:,j])
            tmp[0, j] = theta[0, j] - np.sum(term) * alpha / len(X)
        tmp[0, paratemers-1] = theta[0, paratemers-1] - alpha * np.sum(error) / len(X)
        theta = tmp
        cost[i] = costCalculation(X, y, theta)
    return cost,theta
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta

cols = data2.shape[1]
X = data2.iloc[:,0:cols-1]
y = data2.iloc[:,cols-1:cols]
X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.zeros((1,cols-1))
alpha = 0.01
iters = 1000
#print(X.shape)
#print(y.shape)


initCost = costCalculation(X,y,theta)
print(initCost)
resCost,g = gradientDescent(X,y,theta,alpha,iters)
print(resCost[iters - 1])

#绘图
#data.plot(title='Input Data', kind='scatter', x='Population', y='Profit', figsize = (12, 8))
#plt.show()
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), resCost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()