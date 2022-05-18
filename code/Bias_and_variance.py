#看一下 偏差和方差的区别
#大偏差为欠拟合，大方差为过拟合，过拟合的通过增加样本可以使得其方差趋近稳定，

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
from scipy.io import loadmat

path = "E:\\Apollo\\资料\\Coursera-ML-AndrewNg-Notes-master\\code\\ex5-bias vs variance\\ex5data1.mat"
data = loadmat(path)
def list2ravel(data):
    return map(np.ravel, [data['X'],data['y'],data['Xval'],data['yval'],data['Xtest'],data['ytest']])

X,y,Xval,yval,Xtest,ytest = list2ravel(data)
print(X.shape,y.shape,Xval.shape,yval.shape,Xtest.shape,ytest.shape)

df = pd.DataFrame('')

