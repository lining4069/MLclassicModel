import numpy as np
import matplotlib.pyplot as plt

# 加载数据
def load_dataset(filename):
    with open(filename, 'r') as fr:
        lines = fr.readlines()
    n = len(lines)
    dataX = np.zeros((n, 2))
    dataY = []
    for index, line in enumerate(lines):
        line = line.strip()
        list_of_line = line.split('\t')
        dataX[index, :] = [float(x) for x in list_of_line[:-1]]
        dataY.append(float(list_of_line[-1]))
    return dataX, dataY

# 正规方程的方式计算w
def stand_regres(dataX, dataY):
    dataX = np.mat(dataX)
    dataY = np.mat(dataY).T
    xTx = dataX.T * dataX
    if np.linalg.det(xTx) == 0:
        print("奇异矩阵，不能求逆")
        return
    w = xTx.I * dataX.T * dataY
    return w

def plot_dataset(dataX, dataY, w):
    plt.figure(figsize=(12,8))
    plt.scatter(dataX[:, 0], dataX[:, 1], c=dataY)
    plt.show()

# 用梯度下降的方式求w
def desent_regres(dataX, dataY):
    dataX = np.mat(dataX)
    dataY = np.mat(dataY).T
    w = np.ones((np.shape(dataX)[1], 1))
    alpha = 0.001
    for k in range(500):
        grad = -2 * dataX.T * dataY + 2 * dataX.T * dataX * w
        w = w - alpha * grad
    return w

# 4. 以岭回归的方式求w
# 标准化预处理
def regularize(dataX):
    normX = dataX.copy()
    normX = (dataX - np.mean(dataX[:, 1], 0)) / np.std(dataX[:, 1], 0)
    return normX

def redge_regres(normX, dataY, lam=0.2):
    normX = np.mat(normX)
    dataY = np.mat(dataY).T
    xTx = normX.T * normX
    xTx = xTx + np.eye(np.shape(normX)[1]) * lam
    if np.linalg.det(xTx) == 0:
        print("奇异矩阵，不能求逆")
        return
    w = xTx.I * normX.T * dataY
    return w

if __name__=="__main__":
    filePath = "./ex0.txt"
    datax, dataY = load_dataset(filePath) #type(datax,datay)=(list,list)

