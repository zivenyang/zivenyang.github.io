---
layout:       post
title:        "Win10+Ubuntu16.04双系统安装"
subtitle:     " \"Win+Ubuntu双系统\""
date:         2017-03-26
author:       "Ziven"
header-img:   "img/post-bg-2015.jpg"
header-mask:  0.3
catalog:      true
tags:
    - 双系统
    - Windows
    - Linux
    - Ubuntu
---

> “最近在学Tensorflow，想在虚拟机下用Ubuntu搭环境的，结果发现虚拟机下貌似不能使用GPU加速，只好装了双系统。这篇文章记录了我在安装时的全部操作，双系统使用至今也没有出过什么问题。”

## 准备工作：
&emsp;&emsp;1.一台PC  
&emsp;&emsp;2.一个U盘（8GB以上）  

## Win10安装（已经装好Win10的小朋友们请无视）：  
&emsp;&emsp;1.下载[Win10升级助手](https://www.microsoft.com/zh-cn/software-download/windows10)  
&emsp;&emsp;2.保证系统盘有8GB以上剩余空间  

### 安装步骤（由于安装过程中未记录过程，因此仅用文字叙述要点）：  
&emsp;&emsp;**1.打开下载好的Win10升级助手**  
&emsp;&emsp;&emsp;&emsp;**&bull;选择要安装的版本** *（推荐：简体中文-Windows 10-简体中文  这里Windows 10其实就是专业版，要装其他版本可以在选项中选取）*  
&emsp;&emsp;&emsp;&emsp;**&bull;选择立即升级 or 创建安装介质** *（立即升级直接将本机升级为Win10；创建安装介质需要接U盘或者其他介质，这时会情况U盘所有数据建立安装盘，好处是可以在任意满足Win10安装条件的PC上安装Win10）*  
&emsp;&emsp;&emsp;&emsp;**&bull;等待下载与检查安装包**  
&emsp;&emsp;&emsp;&emsp;**&bull;选择立即升级的即可开始安装系统；选择安装介质安装的需要进入BIOS设置U盘启动** *（ThinkPad可以直接按F12选择启动介质，也可按F1进入BIOS中设置，下图中选择USB HDD）*  
![](/img/in-post/win+ubuntu/BootMenu.jpg)  
&emsp;&emsp;&emsp;&emsp;**&bull;按需选择是否要保留已有的个人文件和设置** *（如果只是想升级或安装Win10建议保留；如果想重装系统则不保留）*  
&emsp;&emsp;&emsp;&emsp;**&bull;按需选择是否删除所有驱动器** *（若选择仅限安装了windows的驱动器，则只对系统盘处理；若选择删除所有驱动器，则将对所有磁盘格式化【不包括安装介质】）*  
&emsp;&emsp;&emsp;&emsp;**&bull;安装ing**  
&emsp;&emsp;&emsp;&emsp;**&bull;按需选择配置+无脑下一步**
&emsp;&emsp;&emsp;&emsp;**&bull;安装成功**
