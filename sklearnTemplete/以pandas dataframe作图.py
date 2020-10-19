#!/usr/bin/env python
# coding: utf-8

# ### 单一列

# In[ ]:


my_df['col_name'].plot(kind='hist', bins=100)


# ### 所有列(交叉)

# In[ ]:


my_df.plot(kind='line', alpha=0.3)


# ### 所有列(分隔)

# In[ ]:


my_df.plot(kind='line', subplots=True, figsize=(8,8))


# ### 散点图

# In[ ]:


my_df.plot(kind='scatter', x='col1', y='col2',
    # c='col_for_color', s=my_df['col_for_size']*10
    );

