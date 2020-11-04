'''
建立基于决策树模型的继承学习bagging集成
'''
# 导入所需包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from math import log
import operator

# 实现构建决策树模型

#定义计算结点的不纯度的函数。基于绝对信息增益
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#定义计算结点不会纯度的函数——计算基尼不纯度
def uniquecounts(rows):
    results = {}
    for row in rows:
        #The result is the last column
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
        return results
def giniimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        # imp+=p1*p1
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp  # 1-imp

#根据指定的特征进行划分数据（从样本集的第axis属性列中选出特征值为给定的value的样本）
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#循环选择对结点的最优的划分   返回在该节点上最优的划分特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # 特征取值的下标
    bestInfoGain = 65535; bestFeature = -1  #最大绝对信息增益，destFeature——处于最大绝对信息增益时的特诊估值的下标
    for i in range(numFeatures):        #每一个i是一个特征列
        featList = [example[i] for example in dataSet]# 第i个特征的特征取值列
        uniqueVals = set(featList)       # 去重操作，获取特征i的取值的种类
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) #通过splitDataSet函数，划分出i特征，特征值为value的样本
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * giniimpurity(subDataSet)
        if (newEntropy < bestInfoGain):
            bestInfoGain = newEntropy
            bestFeature = i
    return bestFeature

#定义处理只有一格特征的情况
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#定义获取某个特征在全部数据集上的特征取值的函数
def get_uniqueVals(dataSet,labels):
    unqueVals={}
    for index in range(len(labels)):
        unqueVals[labels[index]]=set([example[index] for example in dataSet])
    return unqueVals


#创建决策树（ID3）
def createTree(dataSet,labels,unqueValus):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0] #先定义循环的结束条件：结点都是纯节点的时候，，
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  #获取最优的划分特征
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])  #将选出的最优的划分特征，从未划分的特征集合中删除
    uniquefeats=unqueValus[bestFeatLabel]
    for value in uniquefeats:
        subLabels = labels[:]
        second_dataSet=splitDataSet(dataSet, bestFeat, value)
        #处理在训练集中特征样本个数为0的特征
        if(len(second_dataSet)==0):
            featValues = [example[-1] for example in dataSet]
            class_count={}
            for i in featValues:
                if(i not in class_count):
                    class_count[i]=0
                else:
                    class_count[i]+=1
            class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse= True)
            myTree[bestFeatLabel][value]=class_count[0][0]
            continue
        myTree[bestFeatLabel][value] = createTree(second_dataSet,subLabels,unqueValus)
    return myTree

#定义决策树分类模型classify()
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

# 基于实现的CART树，通过集成学习bagging模型。决策方式使用投票方式

# 对随机森林（其实是bagging因为在构架cart树的时候并没有对特征机型随机选择，后期才发现，就不改名称了）进行参数优选
# 参数包括随机森林中的树的个数
def Model_tuning(n_trees,trainData):
    # 生成随机森林中的树，初始化随机森林
    proportion=0.8
    forests=[]
    for i in range(n_trees):
        train_index=np.random.choice([i for i in range(len(trainData))],int((proportion)*len(trainData)))
        dataSet=trainData[train_index,:]
        mytree=createTree(dataSet.tolist(),labels.tolist(),unqiueValus)
        forests.append(mytree)
    # 通过生成的树，对测试集数据进行预测。
    y_predictions_byForest=[]
    for i in range(n_trees):
        y_prediction_byTree=[]
        for testVec in testData:
            y_pred=classify(forests[i],labels.tolist(),list(testVec))
            y_prediction_byTree.append(y_pred)
        y_predictions_byForest.append(y_prediction_byTree)
    # 通过投票的方式来觉得测试集中的一个样本的最终的是预测类别
    results=[]
    y_predictions_byForest=np.array(y_predictions_byForest)
    for i in range(len(testData)):
        counts=np.bincount(y_predictions_byForest[:,i])
        res=np.argmax(counts)
        results.append(res)
    return results

if __name__=="__main__":
    # 获取数据集
    train_file_path = "./data/cmc_train.csv"
    test_file_path = "./data/test.csv"
    labelPath = "./data/test_target.csv"
    realLabels = pd.read_csv(labelPath).iloc[:, 1].values
    train_csv = pd.read_csv(train_file_path)
    labels = train_csv.columns.values[1:-1]
    test_csv = pd.read_csv(test_file_path)
    trainData = train_csv.values[:, 1:]
    testData = test_csv.values[:, 1:]

    # 获取在全部数据集上的各个特征的特征取值
    testDataSet = np.vstack((trainData[:, :-1], testData[:, :]))
    unqiueValus = get_uniqueVals(testDataSet, labels)
    # 随机采样构建决策树
    n_trees = 100
    proportion = 0.8
    forests = []
    for i in range(n_trees):
        dataSet = trainData[np.random.choice([i for i in range(len(trainData))], int((proportion) * len(trainData))), :]
        mytree = createTree(dataSet.tolist(), labels.tolist(), unqiueValus)
        forests.append(mytree)
    # y_predictions_byForest存储n_trees行，len(testData)列的预测结果
    y_predictions_byForest = []
    for i in range(n_trees):
        y_prediction_byTree = []
        for testVec in testData:
            y_pred = classify(forests[i], labels.tolist(), list(testVec))
            y_prediction_byTree.append(y_pred)
        y_predictions_byForest.append(y_prediction_byTree)
    # resluts=[]初始化最终结果
    results = []
    y_predictions_byForest = np.array(y_predictions_byForest)
    for i in range(len(testData)):
        counts = np.bincount(y_predictions_byForest[:, i])
        res = np.argmax(counts)
        results.append(res)
    print(results)
    print(np.mean(realLabels == results))