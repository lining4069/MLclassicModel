'''
实例步骤：
1、准备训练集D
2、对训练集D的属性进行预处理，最大-最小规范化
3、训练集D的属性进行预处理的样本x的前K个近邻， 欧氏距离
4、结合多数投票的分类规则，对x的类别y进行预测。
'''

import numpy as np
import operator

# 读文件获取数据函数
def file_to_matrix(filename):
    with open(filename,'r') as fr:
        lines=fr.readlines()
    n=len(lines)
    dataX=np.zeros((n,3))
    dataY=[]
    for index,line in enumerate(lines):
        list_of_line=line.strip().split('\t')
        dataX[index,:]=[float(i) for i in list_of_line[:-1]]
        if list_of_line[-1]=="didntLike":
            dataY.append(0)
        if list_of_line[-1]=="smallDoses":
            dataY.append(1)
        if list_of_line[-1]=="largeDoses":
            dataY.append(2)
    return dataX,dataY

# 最大最小化处理函数
def auto_norm(dataX):
    max_vector=dataX.max(0)
    min_vector=dataX.min(0)
    n=len(dataX)
    max_mat=np.tile(max_vector,(n,1))
    min_mat=np.tile(max_vector,(n,1))
    normX=(dataX-min_mat)/max_mat
    return normX

# K近邻实现函数
def classify(x, trainX, trainY, k):
    n = len(trainX)
    x_mat = np.tile(x, (n, 1))
    distance = sum(((x_mat - trainX) ** 2), 1) ** 0.5
    index = distance.argsort()
    # count the labels of k's data
    class_count = {}

    for i in range(k):
        i_class = trainY[index[i]]
        class_count[i_class] = class_count.get(i_class, 0) + 1
    class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return class_count[0][0]


# 运行调用
if __name__=="__main__":
    dataX, dataY = file_to_matrix("./data/datingTestSet.txt")
    normX = auto_norm(dataX)
    # 指定k的值
    k=3
    y_pred = classify([-0.55167465, -0.60194861, -0.43736807], normX, dataY, k)



