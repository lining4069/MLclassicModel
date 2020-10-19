#!/usr/bin/env python
# coding: utf-8

# # 降维

# ### 哑编码

# In[ ]:



import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.OneHotEncoder().fit_transform(np.array(X))
result = raw_result.tolist()
# X: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html


# ### 线性判别分析法（LDA）

# In[ ]:




from sklearn.lda import LDA
import numpy as np
from sklearn import preprocessing

matrix = np.array(X)
target = np.array(target)
temp = LDA(n_components=n_components).fit(matrix, target)
coef = temp.coef_
mean = temp.means_
priors = temp.priors_
scalings = temp.scalings_
xbar = temp.xbar_
label = temp.transform(matrix).tolist()
return label, coef.tolist(), mean.tolist(), priors.tolist(), scalings.tolist(), xbar.tolist()

# X: array-like
# target: array-like
# n_components: int
# http://scikit-learn.org/0.15/modules/generated/sklearn.lda.LDA.html


# ### TSNE-t 分布邻域嵌入算法

# In[ ]:




import numpy as np
from sklearn.manifold import TSNE

matrix = np.array(X)
t_sne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
result = t_sne.fit(matrix)
kl_divergence = result.kl_divergence_
label = t_sne.fit_transform(matrix).tolist()

return label, kl_divergence
# X: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html


# ### PCA 主成分分析算法

# In[ ]:




import numpy as np
from sklearn.decomposition import PCA

matrix = np.array(X)
pca = PCA(n_components='mle', svd_solver='auto').fit(matrix)
result = pca.transform(matrix)
label = result.tolist()
return label, pca.components_.tolist(), pca.explained_variance_.tolist(), pca.explained_variance_ratio_.tolist(), pca.mean_.tolist(), pca.noise_variance_



# # 特征选择

# ### 基于树模型

# In[ ]:




import numpy as np
from sklearn import feature_selection
from sklearn.ensemble import GradientBoostingClassifier

matrix = np.array(X)
target = np.array(target)
temp = feature_selection.SelectFromModel(GradientBoostingClassifier()).fit(matrix, target)
indx = temp._get_support_mask().tolist()
scores = get_importance(temp.estimator_).tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

# X: array-like
# target: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html


# ### 基于惩罚值

# In[ ]:




import numpy as np
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression

matrix = np.array(arr0)
target = np.array(target)
temp = feature_selection.SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit(matrix, target)
indx = temp._get_support_mask().tolist()
scores = get_importance(temp.estimator_).tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

# X: array-like
# target: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html


# ### 递归特征消除法

# In[ ]:




import numpy as np
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression

matrix = np.array(arr0)
target = np.array(target)
temp = feature_selection.SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit(matrix, target)
indx = temp._get_support_mask().tolist()
scores = get_importance(temp.estimator_).tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

# X: array-like
# target: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html


# ### 互信息选择法

# In[ ]:




import numpy as np
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression

matrix = np.array(arr0)
target = np.array(target)
temp = feature_selection.SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit(matrix, target)
indx = temp._get_support_mask().tolist()
scores = get_importance(temp.estimator_).tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

# X: array-like
# target: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html


# ### 相关系数选择法

# In[ ]:




import numpy as np
from sklearn import feature_selection
from sklearn.feature_selection import chi2

matrix = np.array(X)
target = np.array(target)
temp = feature_selection.SelectKBest(lambda X, Y: np.array(list(map(lambda x: abs(pearsonr(x, Y)[0]), X.T))), k=k).fit(matrix, target)
scores = temp.scores_.tolist()
indx = temp.get_support().tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

# X: array-like
# target: array-like
# k: int
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html


# ### 卡方检验法

# In[ ]:




import numpy as np
from sklearn import feature_selection
from sklearn.feature_selection import chi2

matrix = np.array(X)
target = np.array(target)
temp = feature_selection.SelectKBest(chi2, k=k).fit(matrix, target)
scores = temp.scores_.tolist()
indx = temp.get_support().tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

# X: array-like
# target: array-like
# k: int
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html


# ### 方差选择

# In[ ]:




