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
&emsp;&emsp;1. 一台PC  
&emsp;&emsp;2. 一个U盘（8GB以上）  

## Win10安装（已经装好Win10的小朋友们请无视）：  
&emsp;&emsp;1. 下载[Win10升级助手](https://www.microsoft.com/zh-cn/software-download/windows10)  
&emsp;&emsp;2. 保证系统盘有8GB以上剩余空间  

### 安装步骤（由于安装过程中未记录过程，因此仅用文字叙述要点）：  
&emsp;&emsp;**1. 打开下载好的Win10升级助手**  
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
&emsp;&emsp;**2. 以迅雷不及掩耳之势进入应用商店，左键上方的头像，选择设置—自动更新应用（关）！！！**  
&emsp;&emsp;**3. 开始菜单—设置—更新和安全—Windows更新—检查更新 **  
&emsp;&emsp;**4. 安装更新ing**  
&emsp;&emsp;**5. Win10安装完成**  

## Ubuntu16.04安装：
### 准备工作：
&emsp;&emsp;**1. 制作安装介质：**  
&emsp;&emsp;&emsp;&emsp;**&bull;下载[Ubuntu16.04IOS镜像文件](https://www.ubuntu.com/download/desktop)** *（付费界面可以选左下角Not now, take me to the download跳过）*  
&emsp;&emsp;&emsp;&emsp;**&bull;下载将镜像烧录至安装介质的软件[UltraISO](https://cn.ultraiso.net/xiazai.html)** *（可以只下试用版）*  
&emsp;&emsp;&emsp;&emsp;**&bull;开始烧录：**  
![](/img/in-post/win+ubuntu/choose.jpg)  

![](/img/in-post/win+ubuntu/writ.jpg)  

![](/img/in-post/win+ubuntu/start.jpg)

&emsp;&emsp;&emsp;&emsp;**&bull;点击写入**  
&emsp;&emsp;&emsp;&emsp;**&bull;写入ing**  
&emsp;&emsp;&emsp;&emsp;**&bull;写入成功**  

&emsp;&emsp;**2. 为即将安装的Ubuntu16.04划分一个空闲区域** *（入门建议40GB左右）*：  
&emsp;&emsp;&emsp;&emsp;**&bull;右击此电脑—管理—存储—磁盘管理—选择一个磁盘—右击选择压缩卷—输入压缩空间量** *（磁盘的1GB=1000MB）* **—确定**  
&emsp;&emsp;&emsp;&emsp;**&bull;已生成未分配的空间**   
![](/img/in-post/win+ubuntu/fenpan.png)  

### 安装步骤：  
&emsp;&emsp;**1. 进入BIOS，关闭快速启动，并设置U盘优先启动**  
![](/img/in-post/win+ubuntu/boot.jpg)  

&emsp;&emsp;**2. Install Ubuntu 16.04**  
&emsp;&emsp;**3. 选择语言**  
&emsp;&emsp;**4. 准备安装Ubuntu** *（按需选择是否要安装更新和第三方软件）*  
&emsp;&emsp;**5. 【关键】选择安装Ubuntu，与其他系统共存** *（这时Ubuntu会自动选择之前分划出的未分配的空间进行自动安装）*  
![](/img/in-post/win+ubuntu/install.png)  

![](/img/in-post/win+ubuntu/install2.png)  

&emsp;&emsp;**6. 选择继续**  
&emsp;&emsp;**7. 选择时区** *（推荐：上海）*  
&emsp;&emsp;**8. 选择键盘** *（推荐：汉语）*  
&emsp;&emsp;**9. 输入用户名密码**  
&emsp;&emsp;**10. 安装ing**  
&emsp;&emsp;**11. 安装完成**  
&emsp;&emsp;**12. 重启后进入Ubuntu**  
&emsp;&emsp;**13. 设置默认Windows10为优先引导：**  
&emsp;&emsp;&emsp;&emsp;**&bull;打开终端**  
&emsp;&emsp;&emsp;&emsp;```$sudo gedit /etc/default/grub```  
&emsp;&emsp;&emsp;&emsp;**&bull;找到** *（就在开头）*  
&emsp;&emsp;&emsp;&emsp;```GRUB_DEFAULT=0```  
&emsp;&emsp;&emsp;&emsp;**这表示默认引导为第一个引导** *（从0开始计数）* ，**一般Windows 10的引导在第5个，因此我们要将其改为：**  
&emsp;&emsp;&emsp;&emsp;```GRUB_DEFAULT=4```   
&emsp;&emsp;&emsp;&emsp;**&bull;更新grub配置文件：**  
&emsp;&emsp;&emsp;&emsp;```$sudo update-grub```

![](/img/in-post/win+ubuntu/grub.jpg)  

&emsp;&emsp;**14. Win10+Ubuntu16.04双系统安装完成**  
&emsp;&emsp;**15. 可以对[Ubuntu进行个性化设置](http://www.bilibili.com/video/av4947544/) **   
&emsp;&emsp;**16. [更改双系统时间差](https://jingyan.baidu.com/article/154b46317b25ca28ca8f41e8.html) **    
##  参考资料：
&emsp;&emsp;&bull;[Win10和Ubuntu16.04双系统安装详解](http://www.jianshu.com/p/16b36b912b02)  
&emsp;&emsp;&bull;[Win7 U盘安装Ubuntu16.04 双系统详细教程](http://blog.csdn.net/coderjyf/article/details/51241919)  
&emsp;&emsp;&bull;[更改win10 ubuntu16.04启动顺序](http://blog.csdn.net/linux_2016/article/details/52348386)  
