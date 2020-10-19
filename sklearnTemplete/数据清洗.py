#!/usr/bin/env python
# coding: utf-8

# ### 多项式数据变换

# In[ ]:


import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.PolynomialFeatures().fit_transform(np.array(X))
result = raw_result.tolist()

# X: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html


# ### 二值化

# In[ ]:


import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.Binarizer(threshold=t).fit_transform(np.array(X))
result = raw_result.tolist()

# X: array-like
# t: float
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html


# ### 归一化

# In[ ]:


import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.Normalizer().fit_transform(np.array(X))
result = raw_result.tolist()

# X: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html


# ### 去除所有重复数据

# In[ ]:


import pandas as pd
# my_df 是一个pandas dataframe
unique_records = my_df.drop_duplicates()


# ### 去除特定列中的重复数据
# 
# 

# In[ ]:


import pandas as pd
# my_df 是一个pandas dataframe
unique_records_for_cols = my_df.drop_duplicates(cols=['col_1', 'col_0'])


# ### 类别转数字

# In[ ]:


# Label1 和 Label2 表示 类别
target = pd.Series(map(lambda x: dict(Label1=1, Label2=0)[x], my_df.target_col.tolist()), my_df.index)
my_df.target_col = target


# ### 特征缩放（特征标准化）

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

