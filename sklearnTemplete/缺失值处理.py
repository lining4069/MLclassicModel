#!/usr/bin/env python
# coding: utf-8

# ### 用一个值填补多列的缺失值
# 

# In[ ]:




my_df[['col_0', 'col_1']] = my_df[['col_0', 'col_1']].fillna(value)


# ### 用一个值填补一列的缺失值

# In[ ]:




my_df['col'] = my_df['col'].fillna(value)


# ### 用最后一个或缺失值的下一个数

# In[ ]:




# - ffill : propagate last valid observation forward to next valid
# - backfill : use NEXT valid observation to fill gap
my_df['col'] = my_df['col'].fillna(method='ffill')


# ### 用一个由聚合得出的值填补缺失值

# In[ ]:




grouped_on = 'col_0' # ['col_1', 'col_1'] # for multiple columns

### Choice of aggregate functions
## On non-NA values in the group
## - numeric choice : mean, median, sum, std, var, min, max, prod
## - group choice : first, last, count
def filling_function(v):
    return v.fillna(v.mean())
                    
my_df['col'] = my_df.groupby(grouped_on)['col'].transform(filling_function)


# ### 去除所有任何缺失值的数据条目

# In[ ]:




records_without_nas = my_df.dropna()


# ### 去除在指定列中带有缺失值的数据条目
# 
# 

# In[ ]:




cols = ['col_0', 'col_1']
records_without_nas_in_cols = my_df.dropna(subset=cols)


# ### 以列为单位审查数据集

# In[ ]:




my_df.info()

