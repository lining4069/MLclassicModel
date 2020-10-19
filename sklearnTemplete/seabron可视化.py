#!/usr/bin/env python
# coding: utf-8

# ### pairplot

# In[ ]:



import seaborn as sns
sns.pairplot(iris,hue='Species')


# ### swarmplot

# In[ ]:




import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.swarmplot(x='species',y='petal_length',data=my_data) # 以iris数据集为例


# ### boxplot

# In[ ]:




import seaborn as sns
sns.boxplot(x='species',y='Petal.Length',data=iris) # 以iris数据集为例


# ### violinplot

# In[ ]:




import seaborn as sns;iris = sns.load_dataset('iris')
sns.violinplot(x='species',y='sepal_length',data=iris) # 以iris数据集为例


# ### displot

# 
# 
# import seaborn as sns, numpy as np,matplotlib.pyplot as plt
# %matplotlib inline
# sns.set(); np.random.seed(0)
# x = np.random.randn(100)
# sns.distplot(x)
# 

# ### heatmap

# In[ ]:




import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
uniform_data = np.random.rand(10, 12)
sns.heatmap(uniform_data)


# ### tsplot

# In[ ]:




import numpy as np; np.random.seed(22)
import seaborn as sns; sns.set(color_codes=True)
x = np.linspace(0, 15, 31);data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
sns.tsplot(data=data)


# ### kdeplot

# In[ ]:




import numpy as np; np.random.seed(10)
import seaborn as sns; sns.set(color_codes=True)
mean, cov = [0, 2], [(1, .5), (.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T
sns.kdeplot(x,y)


# ### countplot

# In[ ]:




import seaborn as sns;titanic= sns.load_dataset('titanic')
sns.countplot(y="deck", hue="class", data=titanic, palette="Greens_d") # 以titanic数据集为例


# ### pointplot

# In[ ]:




import seaborn as sns; iris = sns.load_dataset('iris')
sns.pointplot(x='species',y='petal_width',data=iris) # 以iris数据集为例


# ### barplot

# In[ ]:




import seaborn as sns; titanic=sns.load_dataset('titanic')
sns.barplot(x="sex", y="survived", hue="class", data=titanic)


# ### JoinGrid

# In[ ]:




import seaborn as sns; sns.set(style="ticks", color_codes=True);tips = sns.load_dataset("tips")
g = sns.JointGrid(x="total_bill", y="tip", data=tips) # 以tips数据集为例
g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g.plot(sns.regplot, sns.distplot)


# ### strpplot

# In[ ]:




import seaborn as sns; iris = sns.load_dataset('iris')
sns.stripplot(x='species',y='sepal_length',data=iris)

