title: '[学习笔记]Meta Learning(元学习)'
author: meurice
date: 2021-01-07 20:51:38
tags:
---
## Introduction
　　Meta Learning = Learn to learn  
　　Meta：How to learn a new model
## Meta Learning
　　Meta Learning即“学会学习”，学习了一些task后，机器学会如何去学习新的task，例如机器学习了task1——语音辨识，task2——图像辨识，...，然后给一个新的task（例如文本分类），面对这个新的task，机器能够快速的学习。
![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/0.png)  
　　其步骤和Machine Learning类似，其中不同的是Maching Learning需要找到一个Function **f**，而Meta Learning需要找到的是一个Learning algorithm **F**：  
　　step1. define a set of learning algorithm  
　　step2. goodness of learning algorithm  
　　step3. pick the best learnDing algorithm
![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/1.png)
### step1. Define a set of learning algorithm  
　　首先准备训练资料，其为一堆训练数据D和一堆f的集合，对于每一个task来说，整个流程构成的不再是像Machine Learning中的参数θ，而是构成了一个f，即每当使用新的参数进行初始化时，我们定义了一个新的f。  
![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/2.png)
### step2. Define the goodness of a Function F
　　对于每一个task，都能得到f，并且有损失l。
![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/3.png)  
　　N：N tasks  
　　l^n：Testing loss for task n after training
![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/4.png)
## Omniglot——Few-shot Classification
　　Omniglot 数据集总共包含 5050 个字母。我们通常将这些分成一组包含 3030 个字母的背景（background）集和一组包含 2020 个字母的评估（evaluation）集。  
### N-ways K-shot
　　N-ways K-shot： In each training and test tasks, there are N classes, each has K examples.  
  ![example](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/5.png)

## MAML
　　MAML的基本思想是：对于每一个task中学到的f，其仅决定参数的赋值方式，而不决定模型架构等内容，网络结构是提前固定的。  
### Loss Function
  ![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/6.png)  
  
　　最小化L(Φ)：**Gradient Descent**
  ![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/7.png)
### MAML对比Model Pre-training
　　MAML不在意Φ在train task上的表现如何，而是在意用Φ训练出的参数θ hat ^ n表现如何。  
　　而Model Pre-training希望找到在task1和task2上损失都最小的Φ。
  ![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/10.png)
　　如下图，可能Φ在task1和task2上表现都不太好，但假设拿这个Φ做初始参数，对于task1和task2来说，都能比较容易的找到最佳参数。
  ![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/8.png)
　　而对于Model Pre-training来说，希望找到的是如下图所示的这个Φ，但不保证拿这个Φ去训练后能得到好的θ hat ^ n。
  ![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/9.png)
　　MAML中参数仅被update一次后就被当作最终的参数。其一原因是需要获得的结果参数量过大，多次更新带来的时间成本大大增加。另一原因是MAML的训练目标是训练后得到非常好的Init，希望更新一次后就能得到非常好的效果，一般训练时只更新一次参数，但测试的时候更新多次。
  ![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/11.png)
  ![](https://hexo-img-meurice.oss-cn-beijing.aliyuncs.com/meta_learning/12.png)
## Reptile
## Gradient Descent as LSTM