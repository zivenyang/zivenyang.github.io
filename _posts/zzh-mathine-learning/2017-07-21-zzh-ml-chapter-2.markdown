---
layout:     post
title:      "第2章 模型评估与选择"
date:       1994-11-05
author:     "Ziven"
header-img: "img/in-post/zzh-machine-learning/bg.jpg"
header-mask:  0.3
catalog:      true
tags:
    - 学习笔记
    - 机器学习
---

返回[目录](http://ziven.xin/2017/07/06/zzh-machine-learning-outline/)   
---------------------------------------------------------------
用于查看完整数学公式：手机用户`长按公式`/PC用户`右击公式`出现菜单栏点击`Math Settings`--->`Zoom Trigger`--->`Double-Click`即可`双击放大`文中基于mathjax的公式与图表

## 重点：
### 1. 经验误差与过拟合
错误率(error rate): \\\(E=\frac{a}{m}\\\)  
精度(accuracy): \\(1-E=(1-\frac{a}{m})\times\text{100%}\\)  
\\[
\text{误差(error)}
\begin{cases}
\text{在训练集上的误差}\longrightarrow\text{训练误差(training error)或经验误差(empirical error)}   \\\  
\text{在新样本上的误差}\longrightarrow\text{泛化误差(
  generalization error
  )}
\end{cases}
\\]
过拟合(overfitting):学习器把训练样本学得“太好”，把训练样本自身的一些特点当作了所有潜在样本都会具有的一般性质，这样就会导致泛化性能下降。*即不仅拟合了训练样本的共性特征也过多地拟合了训练样本的个性特征，从而对未在训练样本中出现过新样本的预测能力很弱*   
欠拟合(underfitting):对训练样本的一般性质尚未学好。*即对共性特征都没学好*  

### 2. 评估方法
用于模型选择(model selection):使用一个“测试集(testing set)”来测试学习器对新样本的判别能力，然后以测试集上的“测试误差(testing error)”作为泛化误差的近似。   

而我们只有一个包含有\\(m\\)个样例的数据集\\(\\{(x_{1},y_{1}), (x_{2},y_{2}),\cdots,(x_{m},y_{m})\\}\\)，既要训练又要测试，怎样对学习的泛化误差进行评估并做出选择呢？我们可以通过对\\(D\\)进行适当的处理，从中产生出训练集\\(S\\)和测试集\\(T\\)的方法。    

#### 2.1 留出法(hold-out)
直接将数据集\\(D\\)划分成两个互斥的集合*(训练/测试集的划分要尽可能保持数据分布的一致性，可以使用分层采样(stratified sampling))* ，其中一个集合作为训练集\\(S\\)，另一个集合作为测试集\\(T\\)，即：   
\\[D=S\cup\text{T},\qquad\text{S}\cup\text{T}=\emptyset\\]

\\[
\begin{array}{cc}
\bbox[#A8F,5px,border:1px solid black]
{
  {
  \quad\quad\quad\quad\quad\text{数据集D}\quad\quad\quad\quad\quad
  }
}   \\\   
\downarrow  \\\   
\bbox[white,5px,border:1px solid black]
{
  {
    \\;\quad\quad\text{训练样本S}\quad\quad\\;\\,
  }
}
\bbox[#AFF,5px,border:1px solid black]
{
  {
  \text{测试样本T}
  }
}
\end{array}
\\]   

单次使用留出法得到的估计结果往往不够可靠，在使用留出法时，一般要采用若干次随机划分，重复进行实验评估后取平均值作为留出法的评估结果。
