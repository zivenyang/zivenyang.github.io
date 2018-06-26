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
D(x) & = E[X-E(X)]^2 \\
&=E[X^2-2XE(X)+[E(X)]^2] \\
&=E[X^2]-2E(X)E(X)+[E(X)]^2\\
&=E[X^2]-[E(X)]^2\\
&=1^2\cdot p+0^2\cdot (1-p)-p^2\\
&=p-p^2\\
&=p(1-p)
\end{align}
$$










