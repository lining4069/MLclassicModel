# 知识准备
* numpy
* matplotlib.pyplot
* from matplotlib.font_manager import FontProperties
* random

# 逻辑回归算法原理回顾

## 引入

* 线性回归

    y = w^Tx + b


* 对数线性回归

     lny = w^Tx + b
    
    y = e^{w^Tx + b}


* 广义线性模型
    
     y = g^{-1}( w^Tx + b)


## 逻辑回归-模型

* 二分类任务

     h = \large \frac{1}{1 + e^{-(w^Tx)}} 

     h为样本为1类的概率

## 逻辑回归-策略

* 极大似然估计 \displaystyle\prod_{i=1}^{N}h^{y_i}(1-h)^{1-y_i}

     对数似然,求L(w)$的极大值 L(w) = \displaystyle\sum_{i=1}^{N}[y_ilogh + (1-y_i)log(1-h))]

     求 -L(w)的极小值

## 逻辑回归-算法

* 梯度下降

    梯度 g(w) = [-L(w)]'

    随机梯度下降(1个样本) g(w) = x_i(h_i - y_i)

    小批量梯度下降 
    
    g(w) = \frac{1}{n}\displaystyle\sum_{i=1}^{N}[x_i (h_i - y_i)]
    
    g(w) = \frac{1}{n}[X.T  * (H - Y)]
