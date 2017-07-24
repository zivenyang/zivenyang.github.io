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
错误率(error rate): \\(E=\frac{a}{m}\\)  

精度(accuracy): \\(1-E=(1-\frac{a}{m})\times\text{100%}\\)  

\\[
\text{误差(error)}
\begin{cases}
\text{在训练集上的误差}\longrightarrow\text{训练误差(training error)或经验误差(empirical error)}   \\\  
\text{在新样本上的误差}\longrightarrow\text{泛化误差(generalization error)}
\end{cases}
\\]

过拟合(overfitting):学习器把训练样本学得“太好”，把训练样本自身的一些特点当作了所有潜在样本都会具有的一般性质，这样就会导致泛化性能下降。*即不仅拟合了训练样本的共性特征也过多地拟合了训练样本的个性特征，从而对未在训练样本中出现过新样本的预测能力很弱*   

欠拟合(underfitting):对训练样本的一般性质尚未学好。*即对共性特征都没学好*  

### 2. 评估方法
用于模型选择(model selection):使用一个“测试集(testing set)”来测试学习器对新样本的判别能力，然后以测试集上的“测试误差(testing error)”作为泛化误差的近似。   

而我们只有一个包含有\\(m\\)个样例的数据集\\(\\{(x_{1},y_{1}), (x_{2},y_{2}),\cdots,(x_{m},y_{m})\\}\\)，既要训练又要测试，怎样对学习的泛化误差进行评估并做出选择呢？我们可以通过对\\(D\\)进行适当的处理，从中产生出训练集\\(S\\)和测试集\\(T\\)的方法。    

#### 2.1 留出法(hold-out)
方法：  
直接将数据集\\(D\\)划分成两个互斥的集合*(训练/测试集的划分要尽可能保持数据分布的一致性，可以使用分层采样(stratified sampling))* ，其中一个集合作为训练集\\(S\\)，另一个集合作为测试集\\(T\\)   
\\[D=S\cup\text{T},\qquad\text{S}\cup\text{T}=\emptyset\\]

\\[
\begin{array}{cc}
\bbox[#A8F,5px,border:1px solid black]
{
  {
  \quad\quad\quad\quad\quad\text{数据集}D\quad\quad\quad\quad\quad
  }
}   \\\   
\downarrow  \\\   
\bbox[white,5px,border:1px solid black]
{
  {
    \\;\quad\quad\text{训练样本}S\quad\quad\\;\\,
  }
}
\bbox[#AFF,5px,border:1px solid black]
{
  {
  \text{测试样本}T\\,
  }
}
\end{array}
\\]   

单次使用留出法得到的估计结果往往不够可靠，在使用留出法时，一般要采用若干次随机划分，重复进行实验评估后取平均值作为留出法的评估结果。

#### 2.2 交叉验证法(cross validation)
> 又称\\(k\\)折交叉验证(\\(k\\)-fold cross validation)

方法：
1. 将数据集分为\\(k\\)个大小相似的互斥矩阵即\\(D=D_{1}\cup D_{2}\cup D_{3}\cdots\cup D_{k}, \quad D_{i}\cap D_{j}=\emptyset(i\neq j)\\)。  
每个子集\\(D_{i}\\)都尽可能保持数据分布的一致性。(图中\\(k=10\\))
  ![10折交叉验证示意图1](/img/in-post/zzh-machine-learning/ch2/10-fold_cross_validation1.png)
2. 每次用\\(k-1\\)个子集的并集作为训练集，余下的那个子集作为测试集。
3. 获得\\(k\\)组训练/测试集，从而进行\\(k\\)次训练和测试。
4. 将\\(k\\)个测试结果的均值作为最终结果。
  ![10折交叉验证示意图2](/img/in-post/zzh-machine-learning/ch2/10-fold_cross_validation2.png)
5. 为了减小因样本划分不同而引入的差别，\\(k\\)折交叉验证通常需要随机使用不同的划分重复\\(p\\)次。
6. 最终的评估结果是这\\(p\\)次\\(k\\)折交叉验证结果的均值。

留一法(Leave-One-Out,简称LOO)：
假定数据集包含\\(m\\)个样本，若令\\(k=m\\)，则在\\(k\\)折交叉验证中训练集由\\(m-1\\)个样本组成，而测试集只有一个样本。
* 优点：评估结果准确
* 缺点：当数据集过大时，计算开销大

#### 2.3 自助法(bootstrapping)
方法：  
以自助采样(bootstrap sampling)为基础，假设给定包含\\(m\\)个样本的数据集\\(D\\)，对其拷贝放入\\(D'\\)。
1. 每次随机从\\(D\\)中挑选一个样本，将其`拷贝`放入\\(D'\\)。
2. 将步骤1重复执行\\(m\\)次，此时我们得到了包含\\(m\\)个样本的数据集\\(D'\\)。
3. 将数据集\\(D'\\)用作训练集，\\(D\setminus D'\\)*(“\\(\setminus\\)”为集合减法，\\(D\\)中有而\\(D'\\)中没有的样本)* 用作测试集。  

这样的测试结果亦称包外估计(out-of-bag estimate)。

`自助法`在数据集较小，难以有效划分训练/测试集时很有用；而在初始数据量足够多时，`留出法`和`交叉验证法`更常用一些。
