---
layout:     post
title:      "第1章 绪论"
date:       1994-11-05
author:     "Ziven"
header-img: "img/in-post/zzh-machine-learning/bg.jpg"
header-mask:  0.3
catalog:      true
tags:
    - 学习笔记
    - 机器学习
---
> 返回[目录](http://ziven.xin/2017/07/06/zzh-machine-learning-outline/)  

## 重点：
### 1. 什么是“机器学习(machine learning)”?
> 机器学习所研究的主要内容是关于在计算机上从数据产生“模型(model)”的算法，即“学习算法(learning algorithm)”   

<img src="/img/in-post/zzh-machine-learning/ch1/ml-def.svg" />


> 有了学习算法，我们把经验数据提供给它，它就能基于这些数据产生模型；在面对新的情况时(没有剖开的西瓜)，模型会给我们提供相应的判断。

<img src="/img/in-post/zzh-machine-learning/ch1/ml-def2.svg" />

> 定义：假设用P来评估计算机程序在某任务类T上的性能，若一个程序通过利用经验E在T中任务上获得了性能改善，则我们就说关于T和P，该程序对E进行了学习。

<img src="/img/in-post/zzh-machine-learning/ch1/ml-def3.svg" />

### 2. 基本术语:
<img src="/img/in-post/zzh-machine-learning/ch1/terminology.svg" />
<img src="/img/in-post/zzh-machine-learning/ch1/terminology2.svg" />

> \\(D=\\{x_{1}, x_{2},\cdots, x_{m}\\}\\)\\(\rightarrow\\)包含\\(m\\)个示例的数据集   
\\(x_{i}=(x_{i1}, x_{i2}, \cdots, x_{id})\\)\\(\rightarrow\\)每个示例由\\(d\\)个属性描述，\\(x_i\\)是\\(d\\)维样本空间\\(\mathcal{X}\\)中的一个向量，\\(x_{i}\in\mathcal{X}\\)，其中\\(x_{ij}\\)是\\(x_i\\)在第\\(j\\)个属性上的取值，\\(d\\)称为样本\\(x_i\\)的“维数(dimensionality)”

<img  src="/img\in-post\zzh-machine-learning\ch1\training.svg" />

训练集\\(\\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{m},y_{m})\\}\\)由样例(example)\\(\(x_{i},y_{i}\)\\)组成，其中\\(y_{i}\in\mathcal{Y}\\)，\\(\mathcal{Y}\\)是所有标记(label)的集合，亦称“标记空间(label space)”或“输出空间”。一般地，预测任务是希望通过对训练集\\(\\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{m},y_{m})\\}\\)进行学习，建立一个输入空间\\(\mathcal{X}\\)到输出空间\\(\mathcal{Y}\\)的映射\\(f:\mathcal{X}\mapsto\mathcal{Y}\\)。

<img src="/img/in-post/zzh-machine-learning/ch1/prediction.svg">

<img src="/img/in-post/zzh-machine-learning/ch1/testing.svg">
即得到预测标记\\(y=f(x)\\)

\\[
\text{学习任务}
\begin{cases}
\text{监督学习(supervised learning)} \\
\text{无监督学习(unsupervised learning)}
\end{cases}
\\]


\\[
\begin{cases}
a_1x+b_1y+c_1z=d_1 \\  
a_2x+b_2y+c_2z=d_2 \\   
a_3x+b_3y+c_3z=d_3
\end{cases}
\\]
