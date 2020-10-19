#!/usr/bin/env python
# coding: utf-8

# 伯努利分布

# In[ ]:


import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.MinMaxScaler().fit_transform(np.array(X))
result = raw_result.tolist()

# X: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html


# 正态分布

# In[ ]:


import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.StandardScaler().fit_transform(np.array(X))
result = raw_result.tolist()

# X: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

