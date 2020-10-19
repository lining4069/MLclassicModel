#!/usr/bin/env python
# coding: utf-8

# ## 分类器

# ### 随机森林(Random Forest)

# In[ ]:


# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# criterion: str, 'gini' for Gini impurity (Default) and “entropy” for the information gain.
clf = RandomForestClassifier(criterion='gini')  
# X: array-like, shape = [n_samples, n_features]
# y: array-like, shape = [n_samples] or [n_samples, n_outputs]
clf.fit(np.array(X), np.array(y))


# ### 随机梯度下降(SGD)

# In[ ]:


# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# criterion: str, 'gini' for Gini impurity (Default) and “entropy” for the information gain.
clf = RandomForestClassifier(criterion='gini')  
# X: array-like, shape = [n_samples, n_features]
# y: array-like, shape = [n_samples] or [n_samples, n_outputs]
clf.fit(np.array(X), np.array(y))


# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)

# 数组 X shape:[n_samples, n_features]
# 数组 y shape:[n_samples]


# ### 支持向量积（SVM）

# In[ ]:


# http://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
from sklearn import svm
import numpy as np

# Classifier Option 1: SVC()
clf = svm.SVC()       # kernel = 'linear' or 'rbf' (default) or 'poly' or custom kernels; penalty C = 1.0 (default)
# Option 2: NuSVC()
# clf = svm.NuSVC() 
# Option 3: LinearSVC()
# clf = svm.LinearSVC()     # penalty : str, ‘l1’ or ‘l2’ (default=’l2’)
clf.fit(X, y)                # X shape = [n_samples, n_features], y shape = [n_samples] or [n_samples, n_outputs]

# print(clf.support_vectors_) # get support vectors
# print(clf.support_)         # get indeices of support vectors
# print(clf.n_support_)       # get number of support vectors for each class

mean_accuracy = clf.score(X,y)
print("Accuracy: %.3f"%(mean_accuracy))


# ### 线性支持向量机(SVM)

# In[ ]:


# http://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
from sklearn.svm import SVC
import numpy as np

clf = SVC(kernal='linear', C=1.0, random_state=())
# X: array-like
# y: array-like
clf.fit(np.array(X), np.array(y))


# ### 朴素贝叶斯

# In[ ]:


# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(train_set, train_features).predict(test_set)


# ### 逻辑回归(LR)

# In[ ]:


# http://scikit-learn.org/stable/modules/linear_model.html
from sklearn import linear_model

lr =linear_model.LogisticRegression()  # penalty : str, ‘l1’ or ‘l2’, default: ‘l2’
lr.fit(X, y)


# ### 分类树(Tree)

# In[ ]:


#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier

# criterion = "gini" (CART) or "entropy" (ID3)
clf = DecisionTreeClassifier(criterion = 'entropy' ,random_state = 0)
clf.fit(X,y)


# 随机森林(Random Forest)

# ### AdaBoostClassifier

# In[ ]:


# 决策树集成
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

adaboost_dt_clf = AdaBoostClassifier(
                                    DecisionTreeClassifier(
                                        max_depth=2,   # 决策树最大深度，默认可不输入即不限制子树深度
                                        min_samples_split=20, # 内部结点再划分所需最小样本数，默认值为2，若样本量不大，无需更改，反之增大
                                        min_samples_leaf=5    # 叶子节点最少样本数,默认值为1，若样本量不大，无需更改，反之增大
                                        ),
                                    algorithm="SAMME", # boosting 算法 {‘SAMME’, ‘SAMME.R’}, 默认为后者
                                    n_estimators=200,  # 最多200个弱分类器，默认值为50
                                    learning_rate=0.8  # 学习率，默认值为1
                                     )
adaboost_dt_clf.fit(X,y)


# ### GBDT(梯度加速决策树)

# In[ ]:


# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingClassifier(max_depth=4,   # 决策树最大深度，默认可不输入，即不限制子树深度
                                max_features="auto",  # 寻找最优分割的特征数量，可为int,float,"auto","sqrt","log2",None:
                                n_estimators=100 # Boosting阶段的数量，默认值为100。
                                )
gbdt.fit(X,y)


# ### K-Means

# In[ ]:


# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import KMeans

kmeans = KMeans(
                n_clusters = 2, # 簇的个数，默认值为8
                random_state=0  
                ).fit(X)

print(kmeans.labels_)
print("K Clusters Centroids:\n", kmeans.cluster_centers_)


# ### 使用 keras 做分类

# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

training_epochs = 200 #训练次数，总体数据需要循环多少次
batch_size = 10  

model = Sequential()
input = X.shape[1]
# 隐藏层128
model.add(Dense(128, input_shape=(input,)))
model.add(Activation('relu'))
# Dropout层用于防止过拟合
model.add(Dropout(0.2))
# 隐藏层128
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# 没有激活函数用于输出层，二分类问题，用sigmoid激活函数进行变换，多分类用softmax。
model.add(Dense(1))
model.add(Activation('sigmoid'))
# 使用高效的 ADAM 优化算法以，二分类损失函数binary_crossentropy，多分类的损失函数categorical_crossentropy
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=training_epochs, batch_size=32)


# ## 回归分析

# ### 使用sklearn

# In[ ]:


import sklearn
sklearn.linear_model.LinearRegression()

# http://scikit-learn.org/stable/modules/linear_model.html


# ### 使用Statsmodels的模型

# In[ ]:


import statsmodels.api as sm
results = sm.OLS(y, X).fit()

# y: matrix
# X: constant
# http://www.statsmodels.org/dev/index.html


# ### 用矩阵因式分解计算最小二乘(numpy.linalg.lstsq)
# 
# 

# In[ ]:


import numpy
from numpy import linalg

linalg.lstsq(a, b, rcond=-1)

# a : (M, N) array_like
# b : {(M,), (M, K)} array_like
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html


# ### 更通用的最小二乘极小化(scipy.o

# In[ ]:


import scipy
from scipy import optimize

optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, bounds=(-inf, inf), method=None, jac=None, **kwargs)

# f : callable
# xdata : An M-length sequence or an (k,M)-shaped array for functions with k predictors
# ydata : M-length sequence
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html


# ### 高度专门化的线性回归函数(scipy.stats)
# 
# 

# In[ ]:


import scipy
from scipy import stats

stats.linregress(x, y=none)

# x, y : array_like
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html


# ### 一般的最小二乘多项式拟合

# In[ ]:


import numpy as np
np.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
# x : array_like, shape (M,)
# y : array_like, shape (M,) or (M, K)
# deg : int
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.polyfit.html

