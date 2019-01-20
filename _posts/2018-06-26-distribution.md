---
layout:       post
title:        "手撕概率分布"
subtitle:     " \"机器学习数学基础\""
date:         2018-06-28
author:       "Ziven"
header-img:   "img/in-post/theano-tf/bg.jpg"
header-mask:  0.3
catalog:      true
tags:
    - 机器学习
---
# 随机变量

设$$S=\{e\}$$为随机试验$$E$$的样本空间，如果对于每一个$$e\in S$$，都有一个实数$$X(e)$$与之对应，这样就得到一个定义在$$S$$上的实值单值函数$$X(e)$$，称$$X(e)$$为定义在$$S$$上的一个**随机变量（random variable）**，简记为$$X$$。$$X$$的可能取值可写成$$x_1,x_2, x_3, \cdots,x_k, \cdots$$。     

简单地讲，随机变量可以看作是一个函数映射，它将样本空间中可能的结果映射到实数域。例如，当我们在计算抛一枚硬币正面朝上的概率时，我们的采样空间为$$S=\{正,反\}$$ ，由于“正”和“反”难以用于计算，因此我们可以使用随机变量将其映射为实数，即：    

$$
\begin{align}
X(反)=0    \\
X(正)=1
\end{align}
$$    

****



## 可数与不可数

假如我们可以使用某种算法来列出一个集合中的所有元素，那么我们就称该集合**可数（countable）**，反之称之为**不可数（uncountable）**。       

例如自然数集合$$N=\{1,2,3,\dots\}$$，我们可以迭代地做$$+1$$运算来列出集合中的每一个元素，因此自然数集合$$N$$是可数的。

而对于区间$$[0.1]$$ 上的实数集是不可数的，我们可以使用反证法来证明。假设区间[0,1]上的实数集是可数的，那么一个列表将所有的元素枚举出来，例如：   

$$
\begin{align}
0.1354295\dots\\
0.4294726\dots\\
0.3916831\dots\\
0.9873435\dots\\
0.2918136\dots\\
0.3716182\dots\\
\vdots
\end{align}
$$

我们取以上列表中的对角线元素组成一个新的数$$a=0.121318\dots​$$，然后将$$a​$$中所有的小数位$$+1​$$得到$$a'=0.232429\dots​$$。可以看到$$a​$$与$$a'​$$都在集合$$[0,1]​$$中，然而我们无法通过上述列表中除$$a'​$$以外的元素来还原出$$a'​$$，即使是$$a​$$也不行。因为即使我们将$$a​$$中的前$$n​$$位小数$$+1​$$，$$a'​$$的第$$n+1​$$位以后小数也都与$$a​$$的第$$n+1​$$位以后的小数不同。因此我们永远也无法根据$$a​$$来得到$$a'​$$，所以说区间$$[0,1]​$$上的实数集是不可数的。

需要注意的一点是，**无穷并不代表不可数**，例如自然数集中具有无穷个元素，但它是可数的。

## 离散型随机变量

如果一个随机变量$$X$$的全部可能取值为有限个或者可列多个（即可数的），则称$$X$$为**离散型随机变量（discrete randomvariable）**。       

离散型随机变量的概率分布可用**概率质量函数(probability mass function, PMF)**来描述，即：        

$$
\begin{align}
P(X=x_k)=p_k,\quad k=1, 2,\cdots
\end{align}
$$

如果一个函数$$P​$$是随机变量$$X​$$的PMF，必须满足下面几个条件：      

* $$P​$$的定义域必须是$$X​$$所有可能状态的集合；      

* $$\forall x\in X,0\leqslant P(x)\leqslant 1$$；    

* $$\sum_{x\in X}P(x)=1​$$。     

  

**期望**：   

$$
\begin{align}
E(X)=\sum_{k=0}^{\infty}x_kP(X=x_k)
\end{align}
$$

**方差**：   

$$
\begin{align}
D(X) & = \sum_{0}^{\infty}[x_k-E(X)]^2P(X=x_k)\\
&=E[X-E(X)]^2 \\
&=E[X^2-2XE(X)+[E(X)]^2] \\
&=E[X^2]-2E(X)E(X)+[E(X)]^2\\
&=E[X^2]-[E(X)]^2\\
\end{align}
$$

****



## 连续型随机变量

如果一个随机变量$$X$$的全部可能取值为不可数的，则称$$X​$$为**连续型随机变量（continuous random variable）**，这类随机变量的值域是一个区间（或几个区间的并）。

设连续型随机变量$$X​$$的分布函数为$$F(x)​$$，如果存在非负函数$$f(x)​$$，使得对任意实数$$x​$$，有：  

$$
\begin{align}
F(X)=\int_{-\infty}^{x}f(x)dx
\end{align}    
$$

其中$$f(x)$$为$$X$$的**概率密度函数（probability density function）**。    

如果一个函数$$f(x)​$$是$$X​$$的概率密度函数，必须满足下面几个条件：   

* $$f(x)​$$的值域必须是$$X​$$的所有可能状态的集合；
* $$\forall x\in X,f(x)\geqslant0$$;  
* $$\int f(x)dx=1$$;  



**期望**：     

$$
\begin{align}
E(X)=\int_{-\infty}^{+\infty}xf(x)dx
\end{align}
$$

**方差**：    

$$
\begin{align}
D(X) &= \int_{-\infty}^{+\infty}[x-E(X)]^2f(x)dx\\
&=E[X-E(X)]^2 \\
&=E[X^2-2XE(X)+[E(X)]^2] \\
&=E[X^2]-2E(X)E(X)+[E(X)]^2\\
&=E[X^2]-[E(X)]^2\\
\end{align}
$$

****



## 常见的离散型随机变量的分布

### 两点分布（Bernoulli distribution）

| $$X$$ | $$1$$ |  $$0$$  |
| :---: | :---: | :-----: |
| $$P$$ | $$p$$ | $$1-p$$ |

若随机变量$$X$$的所以有可能取值为0与1，且它的分布律为

$$
\begin{align}
P(X=k)=p^k(1-p)^{1-k},\quad k=0, 1\quad(0\lt p\lt 1)
\end{align}
$$

则有期望：  

$$
E(X)=1\cdot p+0\cdot q=p
$$


方差：   

$$
\begin{align}
D(X) & = E[X^2]-[E(X)]^2\\
&=1^2\cdot p+0^2\cdot (1-p)-p^2\\
&=p-p^2\\
&=p(1-p)
\end{align}    
$$

******



### 二项分布（Binomial distribution）

| $$X$$ |   0   |   1   |   2   |   $$\cdots$$   |   $$n$$   |
| :-----: | :----: | :----: | :----: | :----: | :----: |
|   $$P$$   |   $$C^{0}_{n}p^0(1-p)^{n}$$   |   $$C^{1}_{n}p^1(1-p)^{n-1}$$   |   $$C^{2}_{n}p^2(1-p)^{n-2}$$   |   $$\cdots$$   |   $$C^{n}_{n}p^n(1-p)^{0}$$   |

若随机变量$$X$$的所有可能取值为$$0, 1, 2, \cdots, n$$ ，且它的分布律为       

$$
\begin{align}
P(X=k)=C_{n}^{k}p^{k}(1-p)^{n-k},\quad k=0, 1, 2,\cdots,n\quad (0\lt p\lt 1)
\end{align}
$$


则称随机变量$$X$$服从参数为$$n,p$$ 的二项分布，记为$$X\sim B(n,p)$$       



在$$n$$重伯努利试验中，以$$X$$表示事件$$A$$发生的次数，它的可能取值为$$1, 2, 3, \cdots , n$$，且由二项概率公式有：     

$$
\begin{align}
P(X=k)=P_n(k)=C_{n}^{k}p^{k}(1-p)^{n-k},\quad k=0, 1, 2,\cdots,n
\end{align}
$$

即$$X\sim B(n,p)$$。因此，我们常用二项分布来描述可重复进行独立试验的随机现象。



**方法一：** 由于$$(X=k)$$相互独立且均服从参数为$$p$$的$$0-1$$分布，故期望为：    

$$
\begin{align}
E(X)=\sum^{n}_{k=1}E(X=k)=np
\end{align}
$$

其方差为：  

$$
\begin{align}
D(X)=\sum^{n}_{i=1}D(X=k)=np(1-p)
\end{align}
$$


**方法二：**  

预备知识:    

$$
\begin{align}
(a+b)^{n}=\sum_{i=0}^{n} C_{n}^{i}a^nb^{n-i}
\end{align}
$$


期望为：       

$$
\begin{align}
E(X)&=\sum_{k=0}^{n}kC_{n}^{k}p^k(1-p)^{n-k}\\
&=\sum_{k=0}^{n}k\frac{n!}{k!(n-k)!}p^k(1-p)^{n-k}\\
&=\sum_{k=0}^{n}k\frac{n(n-1)!}{k!(n-k)!}pp^{k-1}(1-p)^{n-k}\\
&=\sum_{k=0}^{n}k\frac{np(n-1)!}{k!(n-k)!}p^{k-1}(1-p)^{n-k}\\
&=np\sum_{k=1}^{n}\frac{(n-1)!}{(k-1)!(n-k)!}p^{k-1}(1-p)^{n-k}\\
&=np\sum_{k=1}^{n}\frac{(n-1)!}{(k-1)![(n-1)-(k-1)]!}p^{k-1}(1-p)^{(n-1)-(k-1)}\\
&=np\sum_{k=1}^{n}C_{n-1}^{k-1}p^{k-1}(1-p)^{(n-1)-(k-1)}\\
&=np(p+1-p)^{n-1}\\
&=np
\end{align}
$$


方差为：  

$$
\begin{align}
D(X)&=E(X^2)-[E(X)]^2\\
&=E(X^2)-n^2p^2
\end{align}
$$


其中：  

| $$X$$ |             0             |              1              |              4              | $$\cdots$$ |          $$n^2$$          |
| :---: | :-----------------------: | :-------------------------: | :-------------------------: | :--------: | :-----------------------: |
|   P   | $$C^{0}_{n}p^0(1-p)^{n}$$ | $$C^{1}_{n}p^1(1-p)^{n-1}$$ | $$C^{2}_{n}p^2(1-p)^{n-2}$$ | $$\cdots$$ | $$C^{n}_{n}p^n(1-p)^{0}$$ |

$$
\begin{align}
E(X^2)&=\sum_{k=0}^{n}k^{2}C_{n}^{k}p^k(1-p)^{n-k}\\
&=\sum_{k=0}^{n}k^2\frac{n!}{k!(n-k)!}p^k(1-p)^{n-k}\\
&=\sum_{k=0}^{n}k(k-1)\frac{n!}{k!(n-k)!}p^k(1-p)^{n-k}+\sum_{k=0}^{n}k\frac{n!}{k!(n-k)!}p^k(1-p)^{n-k}\\
&=\sum_{k=0}^{n}k(k-1)\frac{n!}{k!(n-k)!}p^k(1-p)^{n-k}+E(X)\\
&=\sum_{k=0}^{n}k(k-1)\frac{n!}{k!(n-k)!}p^k(1-p)^{n-k}+np\\
&=\sum_{k=1}^{n}(k-1)\frac{n!}{(k-1)!(n-k)!}p^k(1-p)^{n-k}+np\\
&=\sum_{k=2}^{n}\frac{n!}{(k-2)!(n-k)!}p^k(1-p)^{n-k}+np\\
&=\sum_{k=2}^{n}\frac{n(n-1)(n-2)!}{(k-2)![(n-2)-(k-2)]!}p^2p^{k-2}(1-p)^{(n-2)-(k-2)}+np\\
&=n(n-1)p^2\sum_{k=2}^{n}\frac{(n-2)!}{(k-2)![(n-2)-(k-2)]!}p^{k-2}(1-p)^{(n-2)-(k-2)}+np\\
&=n(n-1)p^2\sum_{k=2}^{n}C_{n-2}^{k-2}p^{k-2}(1-p)^{(n-2)-(k-2)}+np\\
&=n(n-1)p^2(p+1-p)^{n-2}+np\\
&=n(n-1)p^2+np\\
&=n^2p^2-np^2+np\\
\end{align}
$$


故方差：  

$$
\begin{align}
D(X)&=E(X^2)-[E(X)]^2\\
&=E(X^2)-n^2p^2\\
&=n^2p^2-np^2+np-n^2p^2\\
&=np-np^2\\
&=np(1-p)
\end{align}
$$

*********



### 负二项分布

| $$X$$ |           $$r$$           |           $$r+1$$           |           $$r+2$$           | $$\cdots$$ |
| :---: | :-----------------------: | :-------------------------: | :-------------------------: | :--------: |
| $$P$$ | $$C^{r}_{r}p^r(1-p)^{0}$$ | $$C^{r}_{r+1}p^r(1-p)^{1}$$ | $$C^{r}_{r+2}p^r(1-p)^{2}$$ | $$\cdots$$ |

对于一系列独立的成败实验，每次实验成功的概率恒为$$p$$，持续实验直到$$r$$次成功（$$r$$为正整数），则总实验次数$$X$$的概率为：  

$$
\begin{align}
P(X=x;r,p)=C_{x}^{r}p^r(1-p)^{x-r},\quad x\in [r, r+1, r+2,\cdots ,\infty)
\end{align}
$$

由于第$$r$$次实验必定成功，故上式可以写为：  

$$
\begin{align}
P(X=x;r,p)=C_{x-1}^{r-1}p^r(1-p)^{x-r},\quad x\in [r, r+1, r+2,\cdots ,\infty)
\end{align}
$$

若记$$X=k$$为失败次数，则：  

$$
P(X=k;r,p)=C_{k+r-1}^{r-1}p^{r}(1-p)^{k},\quad k\in [0, 1, 2, \cdots, \infty)
$$    

******


### 泊松分布（Poisson distribution）

若随机变量$$X$$的所有可能取值为一切非负整数，且它的分布律为：    

$$
\begin{align}
P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda},\quad k=0, 1, 2, \cdots, \infty \quad \lambda\gt0
\end{align}
$$

则称$$X​$$服从参数为$$\lambda​$$的泊松分布，记为$$X\sim \pi(\lambda)​$$      



预备知识—Taylor展示：    

$$
\begin{align}
f(x)&=\frac{f(x_0)}{0!}+\frac{f'(x_0)}{1!}(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2+\cdots+\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n +R^n(x)\\
e^x&=\frac{e^0}{0!}+\frac{e^0}{1!}(x-0)+\frac{e^0}{2!}(x-0)^2+\cdots+\frac{e^0}{n!}(x-0)^n+R^n(x) \\
&=0+x+\frac{x^2}{2!}+\cdots+\frac{x^n}{n!}+R^n(x)\\
&=x+\frac{x^2}{2!}+\cdots+\frac{x^n}{n!}+R^n(x)\\
&=\sum_{k=1}^{\infty}\frac{x^k}{k!}+R^n(x)\\
1&=xe^{-x}+\frac{x^2}{2!}e^{-x}+\cdots+\frac{x^n}{n!}e^{-x}+R^n(x)e^{-x}\\
&=\sum_{k=1}^{\infty}\frac{x^k}{k!}e^{-x}+R^n(x)e^{-x}
\end{align}
$$      


则其期望为：     

$$
\begin{align}
E(X)&=\sum_{k=0}^{\infty}k\frac{\lambda^{k}}{k!}e^{-\lambda}\\
&=\lambda e^{-\lambda}\sum_{k=0}^{\infty}k\frac{\lambda^{k-1}}{k!}\\
&=\lambda e^{-\lambda}\sum_{k=1}^{\infty}\frac{\lambda^{k-1}}{(k-1)!}\\
&=\lambda e^{-\lambda}e^{\lambda}\\
&=\lambda
\end{align}     
$$

其方差为：    

$$
\begin{align}
D(X)&=E(X^2)-[E(X)]^2\\
&=E(X^2)-\lambda^2\\
&=\sum_{k=0}^{\infty}k^2\frac{\lambda^{k}}{k!}e^{-\lambda}-\lambda^2\\
&=\sum_{k=0}^{\infty}k(k-1)\frac{\lambda^{k}}{k!}e^{-\lambda}+\sum_{k=0}^{\infty}k\frac{\lambda^{k}}{k!}e^{-\lambda}-\lambda^2\\
&=\sum_{k=0}^{\infty}k(k-1)\frac{\lambda^{k}}{k!}e^{-\lambda}+\lambda-\lambda^2\\
&=\lambda\sum_{k=1}^{\infty}(k-1)\frac{\lambda^{k-1}}{(k-1)!}e^{-\lambda}+\lambda-\lambda^2\\
&=\lambda^2\sum_{k=2}^{\infty}\frac{\lambda^{k-2}}{(k-2)!}e^{-\lambda}+\lambda-\lambda^2\\
&=\lambda^2+\lambda-\lambda^2\\
&=\lambda
\end{align}      
$$

故泊松分布的期望与方差都等于参数$$\lambda​$$   



泊松分布与二项分布的关系：   

若$$X\sim B(n,p)$$，当$$n$$比较大而$$p$$又很小时，二项分布近似泊松分布，即：   

$$
\begin{align}
P(X=k)=C_n^kp^k(1-p)^{n-k}\approx \frac{\lambda^k}{k!}e^{-\lambda},\quad k=0, 1, 2, \cdots
\end{align}
$$    

其中，$$\lambda=np$$ 。  



**证明** 设随机变量$$X_n\sim B(n, p_n)$$，且$$\lim_{x \to \infty}np_n=\lambda$$，其中$$\lambda \gt 0 $$为常量，则记$$np_n=\lambda_n$$，即$$p_n=\frac{\lambda_n}{n}$$，得：    
$$
\begin{align}
C_n^kp_n^k(1-p_n)^{n-k}&=\frac{n(n-1)\cdots(n-k+1)}{k!}(\frac{\lambda_n}{n})^k(\frac{n-\lambda_n}{n})^{n-k}\\
&=\frac{n(n-1)\cdots(n-k+1)}{n^k}\frac{\lambda_n^k}{k!}(1-\frac{\lambda_n}{n})^n(1-\frac{\lambda_n}{n})^{-k}\\
&=[1\cdot(1-\frac{1}{n})\cdots(1-\frac{k-1}{n})]\frac{\lambda_n^k}{k!}(1-\frac{\lambda_n}{n})^n(1-\frac{\lambda_n}{n})^{-k}
\end{align}
$$    


其中         

$$
\begin{align}
&\lim_{n\to \infty}\lambda_n^k=\lambda_k,  &\lim_{n \to\infty}(1-\frac{\lambda_n}{n})^n=\lim_{n\to \infty}(1-\frac{\lambda_n}{n})^{-\frac{\lambda_n}{n}\cdot(-\lambda_n)}=e^{-\lambda},\\
&\lim_{n\to \infty}(1-\frac{\lambda_n}{n})^{-k}=1,  &\lim_{n \to\infty}[1\cdot(1-\frac{1}{n})\cdots(1-\frac{k-1}{n})]=1.
\end{align}
$$

故       

$$
\lim_{n\to \infty}P(X_n=k)=\lim_{n\to \infty}C_n^kp^k(1-p)^{n-k}= \frac{\lambda^k}{k!}e^{-\lambda},\quad k=0, 1, 2, \cdots
$$    

****

## 常见的连续型随机变量的分布

### 均匀分布（Uniform Distribution）

若随机变量$$X$$满足的概率密度函数为：  

$$
\begin{align}
f(x)=F'(x)=
\begin{cases}
\frac{1}{b-a},&a\lt x\lt b,\\
0,&\text{其他.}
\end{cases}
\end{align}
$$    

则称$$X$$在$$(a,b)$$上服从**均匀分布(uniform distribution)**，记为$$X\sim U(a, b)$$ 。    

**期望：**    

$$
\begin{align}
E(X) &= \int_{-\infty}^{+\infty}xf(x)dx\\
&=\int_{a}^{b}\frac{1}{b-a}xdx\\
&=\frac{1}{b-a}\cdot \frac{x^2}{2}\bigg| _{a}^{b}\\
&=\frac{1}{b-a}\cdot \frac{(b-a)(b+a)}{2}\\
&=\frac{a+b}{2}
\end{align}
$$

**方差：**   

$$
\begin{align}
D(x)&=E(x^2)-[E(x)]^2\\
&= \int_{-\infty}^{+\infty}x^2f(x^2)dx-(\frac{a+b}{2})^2\\
&=\int_{a}^{b}\frac{1}{b-a}x^2dx-(\frac{a+b}{2})^2\\
&=\frac{1}{b-a}\cdot \frac{x^3}{3}\bigg| _{a}^{b}-(\frac{a+b}{2})^2\\
&=\frac{1}{b-a}\cdot \frac{(b-a)(a^2+ab+b^2)}{3}-(\frac{a+b}{2})^2\\
&=\frac{a^2+ab+b^2}{3}-(\frac{a+b}{2})^2\\
&=\frac{(b-a)^2}{12}
\end{align}
$$     

*****

### 指数分布（Exponential Distribution）

若随机变量$$X$$的概率密度函数为：   

$$
\begin{align}
f(x)=
\begin{cases}
\lambda e^{-\lambda x},&x\gt0,\\
0, &x\le 0.
\end{cases}
\end{align}
$$


其中常数$$\lambda\gt 0$$，则称$$X$$服从参数为$$\lambda$$的**指数分布（exponential distribution）**。    

**期望：**   

$$
\begin{align}
E(X) &= \int_{-\infty}^{+\infty}xf(x)dx\\
&=\int_0^{+\infty}\lambda xe^{-\lambda x}dx\\
&=-\int_0^{+\infty}xde^{-\lambda x}\\
&=-xe^{-\lambda x}\big|_0^{+\infty}+\int_0^{+\infty}e^{-\lambda x}dx\\
&=\int_0^{+\infty}e^{-\lambda x}dx\\
&=-\frac{1}{\lambda}e^{-\lambda x}\big|_0^{+\infty}\\
&=-\frac{1}{\lambda}\cdot(0-1)\\
&=\frac{1}{\lambda}
\end{align}
$$


**方差：**    

$$
\begin{align}
D(x)&=E(x^2)-[E(x)]^2\\
&= \int_{-\infty}^{+\infty}x^2f(x)dx-\frac{1}{\lambda ^2}\\
&=\int_0^{+\infty}\lambda x^2 e^{-\lambda x}dx-\frac{1}{\lambda ^2}\\
&=\int_0^{+\infty}x^2de^{-\lambda x}-\frac{1}{\lambda ^2}\\
&=-\frac{1}{\lambda}(x^2e^{-\lambda x}\big|_0^{+\infty}-\int_0^{+\infty}e^{-\lambda x}dx^2)-\frac{1}{\lambda ^2}\\
&=\frac{1}{\lambda}\int_0^{+\infty}2xe^{-\lambda x}dx-\frac{1}{\lambda ^2}\\
&=\frac{1}{\lambda}\cdot \frac{2}{\lambda}-\frac{1}{\lambda ^2}\\
&=\frac{1}{\lambda ^2}
\end{align}
$$    

*****



### 正态分布（Normal/Gaussian distribution）

若随机变量$$X$$的概率密度函数为：  

$$
\begin{align}
f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}},\quad -\infty \lt x\lt+\infty
\end{align}
$$


其中$$\mu ,\sigma (\sigma \gt0)$$为常数，则称$$X$$服从参数为$$\mu, \sigma ^2$$的**正太分布（normal distribution）**，又称**高斯分布（Gauss distribution）**，记为$$X\sim N(\mu, \sigma^2)$$。     



**预备知识1：**         

令$$t=\frac{x-\mu}{\sigma}$$，则        

$$
\begin{align}
\int_{-\infty}^{+\infty}f(x)dx&=\int_{-\infty}^{+\infty}\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}dx\\
&=\int_{-\infty}^{+\infty}\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(\sigma t)^2}{2\sigma^2}}d(\sigma t + \mu)\\
&=\int_{-\infty}^{+\infty}\frac{1}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt\\
\end{align}
$$


令$$I=\int_{-\infty}^{+\infty}\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx$$，则       

$$
\begin{align}
I^2&=(\int_{-\infty}^{+\infty}\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx)^2\\
&=\int_{-\infty}^{+\infty}\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx \times \int_{-\infty}^{+\infty}\frac{1}{\sqrt{2\pi}}e^{-\frac{y^2}{2}}dy\\
&=\frac{1}{2\pi}\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}e^{-\frac{x^2+y^2}{2}}dxdy
\end{align}
$$

将坐标$$(x,y)$$转换为极坐标$$(r, \theta)$$：     

$$
\begin{cases}
x=r\cdot cos\theta\\
y=r\cdot sin\theta
\end{cases}
$$

故：     

$$
\begin{align}
I^2&=\frac{1}{2\pi}\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}e^{-\frac{x^2+y^2}{2}}dxdy\\
&=\frac{1}{2\pi}\int_{0}^{2\pi}d\theta \int_{0}^{+\infty}r\cdot e^{-\frac{r^2}{2}}dr\\
&=\frac{1}{2\pi}\theta \big|_0^{2\pi}\int_{0}^{+\infty}e^{-\frac{r^2}{2}}d\frac{r^2}{2}\\
&=-e^{-\frac{r^2}{2}}\big |_0^{+\infty}\\
&=-(0-1)\\
&=1
\end{align}
$$  


因为$$f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}\geqslant 0$$，故$$I=\int_{-\infty}^{+\infty}\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx=1$$。       



**预备知识2：**         

令$$f(x)=\frac{x}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$$，因为$$f(x)=-f(-x)$$，故$$f(x)$$为奇函数，故：     

$$
\begin{align}
\int_{-\infty}^{+\infty}f(x)dx=\int_{-\infty}^{+\infty}\frac{x}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx=0
\end{align}
$$


**期望：**   

$$
\begin{align}
E(X) &= \int_{-\infty}^{+\infty}xf(x)dx\\
&=\int_{-\infty}^{+\infty}x\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}dx\qquad\text{令}t=\frac{x-\mu}{\sigma}\\
&=\int_{-\infty}^{+\infty}(\mu+\sigma t)\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(\sigma t)^2}{2\sigma^2}}d(\sigma t + \mu)\\
&=\int_{-\infty}^{+\infty}\frac{1}{\sqrt{2\pi}}(\mu+\sigma t)e^{-\frac{t^2}{2}}dt\\
&=\int_{-\infty}^{+\infty}\frac{\mu}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt+\int_{-\infty}^{+\infty}\frac{\sigma t}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt\\
&=\mu\int_{-\infty}^{+\infty}\frac{1}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt+\sigma\int_{-\infty}^{+\infty}\frac{t}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt\\
&=\mu\cdot 1+\sigma \cdot0\\
&=\mu
\end{align}
$$


**方差：**   

$$
\begin{align}
D(X)&=E([X-E(X)]^2)\\
&=\int_{-\infty}^{+\infty}(x^2-\mu^2)\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}dx\quad \text{令}s=\frac{x-\mu}{\sigma}\\
&=\int_{-\infty}^{+\infty}(s\sigma)^2\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(s\sigma)^2}{2\sigma^2}}d(s\sigma+\mu)\\
&=\sigma^2\int_{-\infty}^{+\infty}s^2\frac{1}{\sqrt{2\pi}}e^{-\frac{s^2}{2}}ds\\
&=-\sigma^2\int_{-\infty}^{+\infty}s\frac{1}{\sqrt{2\pi}}de^{-\frac{s^2}{2}}\\
&=-\frac{\sigma^2}{\sqrt{2\pi}}\cdot se^{-\frac{s^2}{2}}\bigg|_{-\infty}^{+\infty}+\sigma^2\int_{-\infty}^{+\infty}\frac{1}{\sqrt{2\pi}}e^{-\frac{s^2}{2}}ds\\
&=0+\sigma^2 \cdot 1\\
&=\sigma^2
\end{align}
$$     

******



## 总结

|   分布   |            参数             |     期望      |      方差       |
| :------: | :-------------------------: | :-----------: | :-------------: |
| 两点分布 |       $$0\lt p\lt1$$        |     $$p$$     |   $$p(1-p)$$    |
| 二项分布 | $$n \geqslant1,0\lt p\lt1$$ |    $$np$$     |   $$np(1-p)$$   |
| 泊松分布 |      $$\lambda \gt0$$       |  $$\lambda$$  |   $$\lambda$$   |
| 均匀分布 |         $$a \lt b$$         |  $$(a+b)/2$$  | $$(b-a)^2/12$$  |
| 指数分布 |   $$0 \lt \lambda \lt 1$$   | $$1/\lambda$$ | $$1/\lambda^2$$ |
| 正态分布 |     $$\mu,\sigma \gt0$$     |    $$\mu$$    |  $$\sigma^2$$   |

