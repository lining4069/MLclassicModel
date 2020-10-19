#!/usr/bin/env python
# coding: utf-8

# # Pandas基本操作

# ### 按列里的值过滤或去除数据

# In[ ]:




cond = (my_df['col'] == value)

# multiple values
# cond = my_df['col'].isin([value1, value2])

# null value
# cond = my_df['col'].isnull()

# exclude (negate condition)
# cond = ~cond

my_records = my_df[cond]


# ### 基本的数据排列

# In[ ]:




my_df = my_df.sort('col', ascending=False)


# ### 用数据的位置获取数据

# In[ ]:




my_record = my_df.iloc[1]  # !! get the second records, positions start at 0


# ### 计算频率来转置表

# In[ ]:




freqs = my_df.pivot_table(
    rows=["make"],
    cols=["fuel_type", "aspiration"],
    margins=True     # add subtotals on rows and cols
)


# ### 缺失数据

# In[ ]:




df.dropna( ) #丢弃空值
df3.fillna(df3.mean()) #填充空值
df2.replace("a","f") #取代值


# ### 使用某种法则替换列中的值

# In[ ]:




import numpy as np
rules = {
    value: value1,
    value2: value3,
    'Invalid': np.nan  # replace by an true invalid value
}

my_df['col'] = my_df['col'].map(rules)


# ### 使用函数新增列

# In[ ]:




def remove_minus_sign(v):
    return str.replace('-', ' ', max=2)

my_df['col'] = my_df['col'].map(remove_minus_sign)


# ### 分簇

# In[ ]:




import pandas as pd
# Set columns type
my_df['col'] = my_df['col'].astype(np.float64)

# Computations
bins = [0, 100, 1000, 10000, 100000] # 5 binned, labeled 0,1,2,3,4
bins_col = pd.cut(my_df['col'], bins)
bins_col_label = pd.cut(my_df['col'], bins).labels


# ### 从列新建续数据

# In[ ]:




import pandas as pd
dummified_cols = pd.get_dummies(my_df['col']
    # dummy_na=True # to include NaN values
    )


# ### 对多列数据排序

# In[ ]:




my_df = my_df.sort(['col_0', 'col_1'], ascending=[True, False])


# ### 用index获取数据

# In[ ]:




my_record = my_df.loc[label] # !! If the label in the index defines a unique record


# ### 遍历数据字典

# In[ ]:




import pandas as pd
my_dataset = pd.read_csv("path_to_dataset")

i = 0
for my_row_as_dict in my_dataset.iter_rows():
    if i > 10:
        break
    i += 1
    print my_row_as_dict


# ### 获取元组（tuple）的迭代器

# In[ ]:




import pandas as pd
my_dataset = pd.read_csv("path_to_dataset")

i = 0
for my_row_as_tuple in my_dataset.iter_tuples():
    if i > 10:
        break
    i += 1
    print (my_row_as_tuple)


# ### 设置聚合方程转置表

# In[ ]:




stats = my_df.pivot_table(
    rows=["make"],
    cols=["fuel_type", "aspiration"],
    values=["horsepower"],
    aggfunc='max',   # aggregation function
    margins=True     # add subtotals on rows and cols
)


# ### 从dataframe随机抽取N行数据

# In[ ]:




import random
n = 10
sample_rows_index = random.sample(range(len(my_df)), 10)
my_sample = my_df.take(rows)
my_sample_complementary = my_df.drop(rows)


# ### 获取文档

# In[ ]:




# Everywhere
print(my_df.__doc__)
print(my_df.sort.__doc__)

# When using notebook : append a '?' to get help
get_ipython().run_line_magic('pinfo', 'my_df')
get_ipython().run_line_magic('pinfo', 'my_df.sort')


# ### 删除列

# In[ ]:




del my_df['col']


# ### 水平连接数据

# In[ ]:




import pandas as pd
two_dfs_hconcat = pd.concat([my_df, my_df2], axis=1)


# ### 获取单一列数据

# In[ ]:




my_col = my_df['col']


# ### 获取多列数据

# In[ ]:




cols_names = ['col_0', 'col_1']
my_cols_df = my_df[cols_names]


# ### 重命名指定列

# In[ ]:




my_df = my_df.rename(columns = {'col_1':'new_name_1', 'col_2':'new_name_2'})


# ### 重命名所有列

# In[ ]:




my_df.columns = ['new_col_0', 'new_col_1']  # needs the right number of columns


# ### 将dataframe的index设为列名

# In[ ]:




my_df_with_col_index = my_df.set_index("col")
# my_df_col_index.index is [record0_col_val, record1_col_val, ...]


# ### 将dataframe的index重置成标准值

# In[ ]:




my_df_no_index = my_df.reset_index()
# my_df_no_index.index is now [0, 1, 2, ...]


# ### 使用dataframe的index新建列

# In[ ]:




my_df['new_col'] = my_df.index


# ### 垂直连接数据

# In[ ]:




my_df['new_col'] = my_df.index


# ### 将数据及加载为Pandas Dataframe

# In[ ]:


import pandas as pd
# 默认文件类型为csv
my_dataset = pd.read_csv("path_to_dataset")

