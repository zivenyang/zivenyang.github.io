---
layout:     post
title:      "Tensorflow问题记录"
subtitle:   " \"励志成为tf-boy\""
date:       2017-06-30
author:     "Ziven"
header-img: "img/in-post/tf-problems/bg.jpg"
header-mask:  0.3
catalog:      true
tags:
    - Tensorflow
    - 深度学习
---

> “用于记录使用Tensorflow过程中出现的问题。  
PS：配合`Ctrl+f`食用更佳”

###### 错误信息：`ValueError: Variable conv1/weights/Adam/ does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?`
解决办法：【[scope 命名方法](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-12-scope/)】【[Sharing Variables](https://www.tensorflow.org/programmers_guide/variable_scope)】  
* 这是由于共享变量引起的，在LSTM中时常会重复使用`variable_scope`，而在定义变量时若使用`weights = tf.Variables(..., name=...)`这种定义方式，当明明重复时，tensorflow会自动修改`name`以避免重复，而在LSTM中，我们时常会用到重复的变量名，因此我们需要使用`weights = tf.get_variable(...)`来定义变量，同时配合`variable_scope.reuse_variable()`来对变量进行复用。  
而我所遇见的问题是使用了`tf.get_variable_scope.reuse_variable()`将所有的变量都进行了复用，而在我的模型中`CNN`部分的变量是不需要复用的，因此，我将没有使用`tf.get_variable_scope.reuse_variable()`，而是单独对要复用的变量域使用`variable_scope.reuse_variable()`，于是解决了问题。

###### 错误信息：`matplotlib  'ascii' codec can't decode byte 0xe4 in position 0: ordinal not in range(128)`
解决办法：我遇到这个问题是在`matplotlib中`的`plt.title()`中的，`plt.title()`默认ASCII编码，不直接支持中文编码，因此一般使用`plt.title(u'标题')`来解决，而对于变量可以使用一下方式解决：
```python
text = '标题'
plt.title(s=unicode(text, 'utf-8'))
```

###### Tensorflow 正则化方法：
```python
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_2)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_3)
regularizer = tf.contrib.layers.l2_regularizer(scale=5.0/50000)
reg_term = tf.contrib.layers.apply_regularization(regularizer)

loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=z_3)) + reg_term)
```
