#!/usr/bin/env python
# coding: utf-8

# ### 可视化

# In[ ]:




import Matplotlib.pyplot as plt
s.plot()
plt.show()

df2.plot()
plt.show()


# ### 日期

# In[ ]:




df2['Date'] = pd.to_datatime
(df2['Data'])
df2['Date'] = pd.date_range
('2000-1-1', periods=6, freq ='M')
dates = [datetime(2012,5,1), 
datetime(2012,5,2)]
index = pd.DatetimeIndex(dates)
index = pd.date_range
(datetime(2012,2,1)), end, freq='BM'


# ### 结合数据

# In[ ]:




#Merge
pd.merge(data1, data2, 
how='left', on='X1')
pd.merge(data1, data2,
how='right', on='X1')
pd.merge(data1, data2, 
how='inner', on='X1')
pd.merge(data1, data2, 
how='outer', on='X1')

#join
data1.join(data2, how='right')

#concatenate
#vertical
s.append(s2)
#horizontal/vertical
pd.concat([s,s2], axis=1,
 keys=['One','Two'])
pd.concat([data1, data2], 
axis=1, join='inner')


# ### 分组数据

# In[ ]:




#aggregation
df2.groupby(
by=['Data', 'Type']).mean()
df4.groupby(level=0).sum()
df4.groupby(level=0).agg
({'a':lama x:sum(x)/len(x), 
'b':np.sum})

#Transformation
customSum=lamda x:(x+x%2)
df4.groupby
(level=0).transform(customSum)


# ### 复制数据

# In[ ]:




s3.unique() 
#返回唯一值
df2.duplicated('Tyepe') 
#检查重复值
df2.drop_duplicates(
'Type', keep=last'') 
#丢弃重复值
df.index.duplicated() 
#检查索引重复


# ### 高级索引

# In[ ]:




#basic
df3.loc[:, (df3>1).any()] 
#选列 其中任意元素>1
df3.loc[:, (df3>1).all()] 
#选列 其中所有>1
df3.loc[:, df3.isnull().any()] 
#选列 其中含空
df3.loc[:,df3.notnull().all()] 
#选列 其中不含空

df[(df.Country.isin(df2.Type))] 
#寻找相同元素
df3.filter(items=["a","b"]) 
#根据值筛选
df.select(lamda x: not x%5) 
#选特定元素

s.where(s>0) 
#数据分子集

df6.query('second >first') 
#query 数据结构


#selecting
df.set_index('Country') 
#设置索引
df4 = df.reset_index() 
#重置索引
df =df.rename(index = str, 
columns={"Country":"cntry", 
"Capital":"cptl","Population":"ppltn"}) 
#重命名数据结构

#Reindexing
s2 = s.reindex(['a', 'c', 'd', 'e', 'b'])
#Forward Filling
df.reindex(range(4), 
method='ffill')
#Backward Filling
s3 = s.reindex(range(5), 
method='bfill')

#MultiIndexing
arrays = [np.array([1,2,3]), 
np.array([5,4,3])]
df5 = pd.DataFrame(
np.random.rand(3,2), index = arrays)
tuples = list(zip( *arrays))
index = pd.MultiIndex.from_tuples(
tuples, names=['first', 'second'])
df6 = pd.DataFrame(
np.random.rand(3,2), index = index)
df2.set_index(["Data","Type"])


# ### 迭代

# In[ ]:


df.iteritems( ) #列索引 
df.iterrows( ) #行索引


# ### 数据重构

# In[ ]:


#Pivot
df3 = df2.pivot( index = 'Date', 
columns = 'Type', values = 'Value') #行变列

#Pivot Table
df4 = pd.pivot_table( df2,
values='Value',
index = 'Date', 
columns='Type'] #行变列

#Stack/Unstack
stacked = df5.stack( ) 
stacked.unstacked( )

#Melt
pd.melt(df2, id_vars=["Date"], 
value_vars=["Type","Value"], 
value_name="Observations") 
#将列变行


# ### 以元组(tuple)形式按行写入数据

# In[ ]:




import pandas as pd
py_recipe_output = pd.read_csv("data.csv")
writer = py_recipe_output.get_writer()

# t is of the form :
#   (value0, value1, ...)

for t in data_to_write:
    writer.write_tuple(t)


# ### 以字典(dict)形式按行写入数

# In[ ]:




import pandas as pd
py_recipe_output = pd.read_csv("data.csv")
writer = py_recipe_output.get_writer()

# d is of the form :
#   {'col_0': value0, 'col_1': value1, ...}

for d in data_to_write:
    writer.write_row_dict(r)


# ### 使用boolean formula过滤或去除数据

# In[ ]:




import pandas as pd
# single value
cond1 = (my_df['col_0'] == value)
cond2 = (my_df['col_1'].isin([value1, value2]))
# boolean operators :
# - negation : ~  (tilde)
# - or : |
# - and : &
cond = (cond1 | cond2)
my_records = my_df[cond]


# ### 将数据集加载成多个dataframe
# 
# 

# In[ ]:




import pandas as pd
my_dataset = pd.read_csv("data.csv")

for partial_dataframe in my_dataset.iter_dataframes(chunksize=2000):
    # Insert here applicative logic on each partial dataframe.
    pass


# ### 合并有相同名称的列的两个datafame

# In[ ]:




import pandas as pd
# my_df 是一个pandas dataframe
merged = my_df.merge(my_df2,
on='col', # ['col_0', 'col_1'] for many
how="inner",
# suffixes=("_from_my_df", "_from_my_df2"))


# ### 使用条件定位替换值

# In[ ]:




import pandas as pd
# my_df_orig 是一个pandas dataframe
cond = (my_df_orig['col'] != value)
my_df_orig['col'][cond] = "other_value"


# ### 使用目标值的label定位替换值
# 
# 

# In[ ]:




import pandas as pd
# my_df 是一个pandas dataframe
my_df['col'].loc[my_label] = new_value


# ### 使用目标值的位置定位替换

# In[ ]:




import pandas as pd
# my_df 是一个pandas dataframe
my_df[]'col'].iloc[0] = new_value  # replacement for first record


# ### 设置聚合方程进行交叉制表
# 
# 

# In[ ]:




# 交叉制表是统计学中一个将分类数据聚合成列联表的操作
import pandas as pd
# my_df 是一个pandas dataframe
stats =  pd.crosstab(
    rows=my_df["make"],
    cols=[my_df["fuel_type"], my_df["aspiration"]],
    values=my_df["horsepower"],
    aggfunc='max',   # aggregation function
    margins=True     # add subtotals on rows and cols
)


# ### 计算频率进行交叉制表

# In[ ]:




# 交叉制表是统计学中一个将分类数据聚合成列联表的操作
import pandas as pd
# my_df 是一个pandas dataframe
freqs = pd.crosstab(
    rows=my_df["make"],
    cols=[my_df["fuel_type"], my_df["aspiration"]]
)

