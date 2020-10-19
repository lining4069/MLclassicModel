#!/usr/bin/env python
# coding: utf-8

# # 描述性统计信息

# ### 平均数

# In[ ]:




import numpy as np
np.average(np.array(X))

# X: array-like
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.average.html


# ### 列的分位数

# In[ ]:




import pandas as pd
# set columns type
my_df['col'] = my_df['col'].astype(np.float64)

# computations for 4 quantiles : quartiles
bins_col = pd.qcut(my_df['col'], 4)
bins_col_label = pd.qcut(my_df['col'], 4).labels


# ### 多重聚合（组数据）

# In[ ]:




# columns settings
grouped_on = 'col_0'  # ['col_0', 'col_2'] for multiple columns
aggregated_column = 'col_1'

### Choice of aggregate functions
## On non-NA values in the group
## - numeric choice :: mean, median, sum, std, var, min, max, prod
## - group choice :: first, last, count
# list of functions to compute
agg_funcs = ['mean', 'max']


# compute aggregate values
aggregated_values = my_df.groupby(grouped_on)[aggregated_columns].agg(agg_funcs)

# get the aggregate of group
aggregated_values.ix[group]


# ### 用户定义方程（组数据）

# In[ ]:




# columns settings
grouped_on = ['col_0']
aggregated_columns = ['col_1']

def my_func(my_group_array):
    return my_group_array.min() * my_group_array.count()

## list of functions to compute
agg_funcs = [my_func] # could be many

# compute aggregate values
aggregated_values = my_df.groupby(grouped_on)[aggregated_columns].agg(agg_funcs)


# ### 在聚合的dataframe上使用用户定义方程

# In[ ]:




# columns settings
grouped_on = ['col_0']
aggregated_columns = ['col_1']

def my_func(my_group_array):
    return my_group_array.min() * my_group_array.count()

## list of functions to compute
agg_funcs = [my_func] # could be many

# compute aggregate values
aggregated_values = my_df.groupby(grouped_on)[aggregated_columns].agg(agg_funcs)


# ### 相关性

# In[ ]:




my_df.corr()


# ### 移动平均数

# In[ ]:




import numpy as np

ret = np.cumsum(np.array(X), dtype=float)
ret[w:] = ret[w:] - ret[:-w]
result = ret[w - 1:] / w

# X: array-like
# window: int


# ### 中位数

# In[ ]:




import numpy as np
np.median(np.array(X))

# X: array-like
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.median.html


# ### 一些列的单一属性（如最大最大值，列的类型为数值型）

# In[ ]:




my_df["col"].max() # [["col_0", "col_1"]] 多字段


# ### 组数据的基本信息

# In[ ]:




# columns settings
grouped_on = 'col_0'  # ['col_0', 'col_1'] for multiple columns
aggregated_column = 'col_1'

### Choice of aggregate functions
## On non-NA values in the group
## - numeric choice : mean, median, sum, std, var, min, max, prod
## - group choice : first, last, count
## On the group lines
## - size of the group : size
aggregated_values = my_df.groupby(grouped_on)[aggregated_column].mean()
aggregated_values.name = 'mean'

# get the aggregate of group
aggregated_values.ix[group]


# ### 数据组的遍历

# In[ ]:




# columns settings
grouped_on = 'col_0'  # ['col_0', 'col_1'] for multiple columns

grouped = my_df.groupby(grouped_on)

i = 0
for group_name, group_dataframe in grouped:
    if i > 10:
        break
    i += 1
    print(i, group_name, group_dataframe.mean())  ## mean on all numerical columns


# ### 协方差

# In[ ]:




my_df.cov()


# ### K平均数算法

# In[ ]:




import numpy as np
from sklearn.cluster import KMeans

k_means = KMeans(k).fit(np.array(X))
result = k_means.labels_
label = result.tolist()
return label, k, k_means.cluster_centers_.tolist(), k_means.inertia_

# k: int, k>=2
# X: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


# ### 最大互信息数

# In[ ]:




import numpy as np

matrix = np.transpose(np.array(X)).astype(float)
mine = MINE(alpha=0.6, c=15, est="mic_approx")
mic_result = []
for i in matrix[1:]:
    mine.compute_score(t_matrix[0], i)
    mic_result.append(mine.mic())
return mic_result



# ### 皮尔森相关系数

# In[ ]:




import numpy as np 

matrix = np.transpose(np.array(X))
np.corrcoef(matrix[0], matrix[1])[0, 1]

# X: array-like
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.corrcoef.html



# ### 标准差

# In[ ]:




import numpy as np
np.std(np.array(X))

# X: array-like
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.std.html


# ### 方差

# In[ ]:




import numpy as np

np.var(np.array(X))

# X: array-like
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.var.html


# ### 行列数

# In[ ]:





# ### 以频率降序排列

# In[ ]:





# ### 以列值排序

# In[ ]:





# ### 所有列（列的数据类型为数值型）

# In[ ]:





# ### 所有列的单一属性（如最大值，列的数据类型为数值型）

# In[ ]:





# ### 平均数

# In[ ]:





# ### 获得类别字段的频数

# In[ ]:





# ### 查看缺失情况

# In[ ]:





# ### 查看字段包含unique值的程度

# In[ ]:





# In[ ]:





# In[ ]:




