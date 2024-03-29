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

返回[目录](http://ziven.xin/2017/07/06/zzh-machine-learning-outline/)   
---------------------------------------------------------------
用于查看完整数学公式：手机用户`长按公式`/PC用户`右击公式`出现菜单栏点击`Math Settings`--->`Zoom Trigger`--->`Double-Click`即可`双击放大`文中基于mathjax的公式与图表

## 重点：
### 1. 什么是“机器学习(machine learning)”?
机器学习所研究的主要内容是关于在计算机上从数据产生“模型(model)”的算法，即“学习算法(learning algorithm)”   

\\[
\bbox[yellow,5px,border:1px solid black]
{
  {
  \text{经验数据}
  }
}
\longrightarrow
\bbox[white,5px,border:1px solid black]
{
  {
    \text{学习算法(learning algorithm)}
  }
}
\longrightarrow
\bbox[#AF0,5px,border:1px solid black]
{
  {
    \text{模型(model)}
  }
}
\\]


有了学习算法，我们把经验数据提供给它，它就能基于这些数据产生模型；在面对新的情况时(没有剖开的西瓜)，模型会给我们提供相应的判断。

\\[
\bbox[yellow,5px,border:1px solid black]
{
  {
  \text{没有剖开的西瓜}
  }
}
\longrightarrow
\bbox[#AF0,5px,border:1px solid black]
{
  {
    \text{模型(model)}
  }
}
\longrightarrow
\bbox[#F8A,5px,border:1px solid black]
{
  {
    \text{好瓜}
  }
}
\\]


定义：假设用P来评估计算机程序在某任务类T上的性能，若一个程序通过利用经验E在T中任务上获得了性能改善，则我们就说关于T和P，该程序对E进行了学习。

\\[\bbox[20px,border:1px solid black]
{\{\text{任务类\\(\mathtt{T}\\)中的任务} \\\  
\bbox[yellow,5px,border:1px solid black]
{
  {
  \text{经验}\mathtt{E}
  }
}
\-{\text{学习}}\rightarrow
\bbox[5px,border:1px solid black]
{
  {
    \text{程序}
  }
}
\longrightarrow
\bbox[#AFF,5px,border:1px solid black]
{
  {
    \text{性能}\mathtt{P}\uparrow
  }
}
\}
}
\\]
### 2. 基本术语:
<img src="/img/in-post/zzh-machine-learning/ch1/terminology.svg" />
<img src="/img/in-post/zzh-machine-learning/ch1/terminology2.svg" />

\\(D=\\{x_{1}, x_{2},\cdots, x_{m}\\}\\)\\(\rightarrow\\)包含\\(m\\)个示例的数据集   
\\(x_{i}=(x_{i1}, x_{i2}, \cdots, x_{id})\\)\\(\rightarrow\\)每个示例由\\(d\\)个属性描述，\\(x_i\\)是\\(d\\)维样本空间\\(\mathcal{X}\\)中的一个向量，\\(x_{i}\in\mathcal{X}\\)，其中\\(x_{ij}\\)是\\(x_i\\)在第\\(j\\)个属性上的取值，\\(d\\)称为样本\\(x_i\\)的“维数(dimensionality)”

训练(training)或学习(learning):
\\[
\bbox[yellow,5px,border:1px solid black]
{
  {
  \text{训练数据(training data)}
  }
}
\longrightarrow
\bbox[white,5px,border:1px solid black]
{
  {
    \text{假设(hypothesis)}
  }
}
\-\text{逼近}\rightarrow
\bbox[#F8A,5px,border:1px solid black]
{
  {
    \text{“真相”或“真实”(ground truth)}
  }
}
\\]

训练集\\(\\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{m},y_{m})\\}\\)由样例(example)\\(\(x_{i},y_{i}\)\\)组成，其中\\(y_{i}\in\mathcal{Y}\\)，\\(\mathcal{Y}\\)是所有标记(label)的集合，亦称“标记空间(label space)”或“输出空间”。一般地，预测任务是希望通过对训练集\\(\\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{m},y_{m})\\}\\)进行学习，建立一个输入空间\\(\mathcal{X}\\)到输出空间\\(\mathcal{Y}\\)的映射\\(f:\mathcal{X}\mapsto\mathcal{Y}\\)。

预测(prediction):
\\[
\bbox[yellow,5px,border:1px solid black]
{
  {
  \text{训练集\\(\\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{m},y_{m})\\}\\)}
  }
}
\-\text{学习}\rightarrow
\bbox[white,5px,border:1px solid black]
{
  {
  f:\mathcal{X}\mapsto\mathcal{Y}
  }
}
\\]

测试(testing):
\\[
\bbox[#A8F,5px,border:1px solid black]
{
  {
  \text{测试样本(testing sample)}
  }
}
\longrightarrow
\bbox[white,5px,border:1px solid black]
{
  {
  f:\mathcal{X}\mapsto\mathcal{Y}
  }
}
\longrightarrow
y
\\]
即得到预测标记\\(y=f(x)\\)

\\[
\text{学习任务}
\begin{cases}
\text{监督学习(supervised learning)}  \begin{cases}
\text{分类(classification)} \begin{cases}
\text{二分类(binary classification)} \longrightarrow \mathcal{Y}=\\{-1,+1\\} \\\  
\text{多分类(multi-class classification)} \longrightarrow |\mathcal{Y}|>2
\end{cases} \\\  
\text{回归(regression)} \longrightarrow \mathcal{Y}=\mathbb{R}, \mathbb{R}\text{为实数集}
\end{cases} \\\  
\text{无监督学习(unsupervised learning)} \begin{cases}
\text{聚类(cluster)} \longrightarrow \text{没有}\mathcal{Y} \\\  
\vdots
\end{cases}
\end{cases}
\\]

学得模型适用于新样本的能力，称为“泛化(generalization)”能力
通常假设样本空间中全体样本服从一个未知“分布(distribution)”\\(\mathcal{D}\\)，我们获得的每个样本都是独立地从这个样本上采样获得的，即“独立同分布(independent and identically distributed，简称\\(i.i.d.\\))”。一般而言，训练样本越多，我们得到关于\\(\mathcal{D}\\)的信息越多，这样就越有可能通过学习获得具有强泛化能力的模型。

\\[\bbox[10px,border:1px solid black]
{
\text{训练样本}\uparrow\space\longrightarrow\space\text{关于\\(\mathcal{D}\\)的信息}\uparrow\space\longrightarrow\space\text{模型泛化能力}\uparrow
}
\\]

### 3. 假设空间：
我们可以把学习过程看作一个在所有假设(hypothesis)组成的空间中进行搜索的过程，搜索目标是找到与训练集“匹配(fit)”的假设，即能够将训练集中的瓜判断正确的假设。
\\[
\begin{array}{ccccc}
\text{编号} & \text{色泽} & \text{根蒂} & \text{敲声} & \text{好瓜} \\\   
\hline
1 & \text{青绿} & \text{蜷缩} & \text{浊响} & \text{是} \\\   
2 & \text{乌黑} & \text{蜷缩} & \text{浊响} & \text{是} \\\   
3 & \text{青绿} & \text{硬挺} & \text{清脆} & \text{否} \\\   
4 & \text{乌黑} & \text{稍蜷} & \text{沉闷} & \text{否}
\end{array}
\\]
假设的表示一旦确定，假设空间及规模大小就确定了。这里我们的假设空间由形如“\\((\text{色泽=?})\wedge(\text{根蒂=?})\wedge(\text{敲声=?})\\)”的可能取值所形成的假设组成。

<img src="/img/in-post/zzh-machine-learning/ch1/hypothesis-space.svg" />
\\[4\times3\times3=36\text{种假设}\\]
\\[\varnothing\quad\text{也为1种假设}\\]
\\[\text{因此共有}36+1=37\text{种假设}\\]

需注意的是，现实问题中我们常面临很大的假设空间，单学习过程是基于有限样本训练集进行的，因此，可能有多个假设与训练集一致，即存在着一个与训练集一致的“假设集合”，我们称之为“版本空间(version space)”。
上表中所对应的版本空间(version space)：
<img src="/img/in-post/zzh-machine-learning/ch1/version-space.svg" />
这样上文假设空间的37种假设就被挑选出版本中的3种假设了，而版本空间中的任意一种假设都能与训练集中的各个训练样本匹配。

### 4. 归纳偏好：
虽然然版本空间中的各个假设都能与各个训练样本匹配，但与它们对应的模型在面临新的样本选择时却会产生不同的输出。
\\[
\begin{array}{cc}
\bbox[#A8F,5px,border:1px solid black]
{
  {
  \text{(色泽=青绿；根蒂=蜷缩；敲声=沉闷)}
  }
}  &
\bbox[#A8F,5px,border:1px solid black]
{
  {
  \text{(色泽=青绿；根蒂=蜷缩；敲声=沉闷)}
  }
}  \\\   
\downarrow  & \downarrow \\\   
\bbox[white,5px,border:1px solid black]
{
  {
  \text{(好瓜)}\leftrightarrow\text{(色泽=\*)}\wedge\text{(根蒂=蜷缩)}\wedge\text{(敲声=\*)}
  }
} &
\bbox[white,5px,border:1px solid blue]
{
  {
  \text{(好瓜)}\leftrightarrow\text{(色泽=\*)}\wedge\text{(根蒂=\*)}\wedge\text{(敲声=浊响)}
  }
} \\\  
\downarrow & \downarrow \\\  
\text{好瓜} & \color{red}{不是好瓜}
\end{array}
\\]
那么有没有一般性的原则来引导算法确立“正确的”偏好呢？  
“奥卡姆剃刀(Occam's razor)”是一种常用的、自然科学研究中最基本的原则，即`“若有多个假设与观察一致，则选择最简单的那个”`  
但是“没有免费的午餐”定理(No Free Lunch Theorem,简称NFL定理)却表明无论一个学习算法多聪明或者多笨拙，它们的期望性能是一样的。  
不过，NFL定理有一个重要前提：所有“问题”出现的机会相同、或所有问题同等重要。但实际情形并不是这样。很多时候，我们只关注自己正在试图解决的问题(例如某个具体应用任务)，希望为它找到一个解决方案，至于这个解决方案在别的问题、甚至在相似的问题上是否为好的方案，我们并不关心。   
所以，NFL定理最重要的寓意，是让我们清楚地认识到，`脱离具体问题，空泛地谈论“什么算法更好”毫无意义`。

<!-- ### 习题：
> 1.1 表1.1中若只包含编号为1和4的两个样例，试给出相应的版本空间。

即：
\\[
\begin{array}{ccccc}
\text{编号} & \text{色泽} & \text{根蒂} & \text{敲声} & \text{好瓜} \\\   
\hline
1 & \text{青绿} & \text{蜷缩} & \text{浊响} & \text{是} \\\    
4 & \text{乌黑} & \text{稍蜷} & \text{沉闷} & \text{否}
\end{array}
\\]

因此版本空间是：
\\[
\bbox[white,5px,border:1px solid black]
{
  \text{(色泽=青绿；根蒂=蜷缩；敲声=浊响)}
}
\\]

> 1.2 与使用单个合取式来进行假设表示相比，使用“析合范式”将使得假设空间具有更强的表示能力。例如  
\\[
\text{好瓜}\leftrightarrow\bigl((\text{色泽=\*})\wedge(\text{根蒂=蜷缩})\wedge(\text{敲声=\*})\bigr) \\\
\qquad\qquad\lor\bigl((\text{色泽=乌黑})\land(\text{根蒂=\*})\land(\text{敲声=沉闷})\bigr),
\\]
会把“\\((\text{色泽=青绿})\wedge(\text{根蒂=蜷缩})\wedge(\text{敲声=清脆})\\)”以及“\\((\text{色泽=乌黑})\land(\text{根蒂=硬挺})\land(\text{敲声=沉闷})\\)”都分类为“好瓜”。若使用最多包含\\(k\\)个合区范式的析合范式来表达表1.1西瓜分类问题的假设空间，试估算共有多少种可能的假设。

\\[
\begin{array}{ccccc}
\text{编号} & \text{色泽} & \text{根蒂} & \text{敲声} & \text{好瓜} \\\   
\hline
1 & \text{青绿} & \text{蜷缩} & \text{浊响} & \text{是} \\\   
2 & \text{乌黑} & \text{蜷缩} & \text{浊响} & \text{是} \\\   
3 & \text{青绿} & \text{硬挺} & \text{清脆} & \text{否} \\\   
4 & \text{乌黑} & \text{稍蜷} & \text{沉闷} & \text{否}
\end{array}
\\]
\\[
\text{好瓜}\leftrightarrow\bigl((\text{色泽=\*})\wedge(\text{根蒂=蜷缩})\wedge(\text{敲声=\*})\bigr) \\\
\qquad\qquad\lor\bigl((\text{色泽=乌黑})\land(\text{根蒂=\*})\land(\text{敲声=沉闷})\bigr) \\\
\\]
> 1.3 若数据集包含噪声，则假设空间中有可能不存在与所有训练样本都一致的假设。在此情形下，试设计一种归纳偏好用于假设选择。

> 1.5 试述机器学习能在互联网搜索的哪些环节起什么作用。 -->
