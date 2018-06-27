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

## Bernoulli distribution

已知随机变量$$X$$的分布为  

| $$X$$ | $$1$$ |  $$0$$  |
| :---: | :---: | :-----: |
| $$P$$ | $$p$$ | $$1-p$$ |

则有期望：  

$$
E(X)=1\cdot p+0\cdot q=p
$$


方差：  

$$
\begin{align}
D(X) & = E[X-E(X)]^2 \\
&=E[X^2-2XE(X)+[E(X)]^2] \\
&=E[X^2]-2E(X)E(X)+[E(X)]^2\\
&=E[X^2]-[E(X)]^2\\
&=1^2\cdot p+0^2\cdot (1-p)-p^2\\
&=p-p^2\\
&=p(1-p)
\end{align}
$$

## Binomial distribution

| $$X$$ |   0   |   1   |   2   |   $$\cdots$$   |   $$n$$   |
| :-----: | :----: | :----: | :----: | :----: | :----: |
|   $$P$$   |   $$C^{0}_{n}p^0(1-p)^{n}$$   |   $$C^{1}_{n}p^1(1-p)^{n-1}$$   |   $$C^{2}_{n}p^2(1-p)^{n-2}$$   |   $$\cdots$$   |   $$C^{n}_{n}p^n(1-p)^{0}$$   |



设随机变量$$X$$服从参数为$$n,p$$二项分布，即$$X\sim B(n,p)$$  

设$$X_i$$为第$$i$$次试验中事件$$A$$发生的次数，$$i=1, 2, 3, \cdots , n$$则  
$$
\begin{align}
X=\sum^{n}_{i=1}{X_i}
\end{align}
$$

其中$$X_i=b(i;n,p)=C^{i}_{n}p^i(1-p)^{n-i}$$  

**方法一：** 由于$$X_i$$相互独立且均服从参数为$$p$$的$$0-1$$分布，故期望为：  
$$
\begin{align}
E(X)=\sum^{n}_{i=1}E(X_i)=np
\end{align}
$$

其方差为：  
$$
\begin{align}
D(X)=\sum^{n}_{i=1}D(X_i)=np(1-p)
\end{align}
$$

**方法二：**  

预备知识:  
$$
\begin{align}
X_i=b(i;n,p)=C^{i}_{n}p^i(1-p)^{n-i}\\
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





## 负二项分布

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


## Poisson distribution

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



设$$X\sim \pi(\lambda)$$，且    
$$
\begin{align}
P\{X=k\}=\frac{\lambda^k}{k!}e^{-\lambda},\quad k=0, 1, 2, \cdots, \infty \quad \lambda\gt0
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


故泊松分布的期望与方差都等于参数$$\lambda$$

