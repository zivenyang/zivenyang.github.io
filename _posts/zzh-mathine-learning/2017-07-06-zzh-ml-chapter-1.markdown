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

![ml-def2](https://cdn.rawgit.com/zivenyang/draw/26a18a7b/zzh-ml/ch1/ml-def2.svg)

> 有了学习算法，我们把经验数据提供给它，它就能基于这些数据产生模型；在面对新的情况时(没有剖开的西瓜)，模型会给我们提供相应的判断。

![ml-def3](https://cdn.rawgit.com/zivenyang/draw/7622f904/zzh-ml/ch1/ml-def3.svg)

> 定义：假设用P来评估计算机程序在某任务类T上的性能，若一个程序通过利用经验E在T中任务上获得了性能改善，则我们就说关于T和P，该程序对E进行了学习。

![ml-def](https://cdn.rawgit.com/zivenyang/draw/b64ab860/zzh-ml/ch1/ml-def.svg)

### 2. 基本术语:
![terminology](https://cdn.rawgit.com/zivenyang/draw/c8417e3e/zzh-ml/ch1/terminology.svg)
![terminology2](https://cdn.rawgit.com/zivenyang/draw/573d142d/zzh-ml/ch1/terminology2.svg)

\\(D=\{x_{1}, x_{2}, \cdots, x_{m}\}\\)\\(\rightarrow\\)包\\(m\\)个示例的数据集   
\\(x_{i}=(x_{i1}, x_{i2}, \cdots, x_{id})\\)\\(\rightarrow\\)每个示例由\\(d\\)个属性描述，\\(x_i\\)是\\(d\\)维样本空间\\(\mathcal{X}\\)中的一个向量，\\(x_{i}\in\mathcal{X}\\)，其中\\(x_{ij}\\)是\\(x_i\\)在第\\(j\\)个属性上的取值，\\(d\\)称为样本\\(x_i\\)的“维数(dimensionality)”
