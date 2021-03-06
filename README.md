# Pytorch 文本(长文本)分类任务 Demo

​	本demo是在学习和练习文本分类的过程中记录下来的一个demo。主要是温习和练习一些基本的文本分类神经网络。文档里面实现的方法基本都有详细的说明，主要是方便后期查看。

​	Demo实现的是长文本分类问题，主要是思想是将长文本保留，在embedding后将每个句子整合成一个向量（采用每个词相加，或者最大池化的方式进行处理，本demo采用的是最大池化），然后进行正常的文本分类操作。

​		

​	本Demo主要基于**[ Pytorch_Text_Classification_Demo](https://github.com/CLOVEXCWZ/Pytorch_Text_Classification_Demo)**的基础进行修改，两个demo的区别就在于一个是处理短文本一个是处理长文本，其他的处理都是一样的。




**注意**
- 项目主要是练习，所以参数方面并没有过多的调整。



## **数据集**

** [搜狗新闻数据语料原地址](http://www.sogou.com/labs/resource/cs.php) **

[搜狗新闻数据语料地址](http://www.sogou.com/labs/resource/cs.php)  这里下载处理的是 347M的简版

[处理后的数据集百度云盘下载地址](https://pan.baidu.com/s/1xfWDTI_fXqKB2mFs-9MiLw)  提取码：mbt3



​	新闻语料中主要有包含多个类别，由于考虑到样本量和样本均衡情况，只选取前4种样本量比较多且较为均衡的类别作为项目的数据集。

**处理后数据情况：更为详细的数据情况请移步到datas/sougouNew/下面的Readme文件中查看**



**训练集和验证集样本情况**

注意：训练集和测试集做过缺失值处理。经过处理后的文本每行包含一个样本，文本和标签用'\t'分开，前面为文本，标签在后面。



训练集样本数量：263505     验证集样本数量：30000

| 类别     | 训练集样本数量 | 验证集样本数量 |
| -------- | -------------- | -------------- |
| sports   | 73715          | 8435           |
| news     | 74048          | 8568           |
| house    | 62231          | 6960           |
| business | 53511          | 6037           |

**训练集中数据情况**

|      | 每篇句子数量 | 每篇长度     | 所有句子长度 |
| ---- | ------------ | ------------ | ------------ |
| mean | 11.671706    | 797.463122   | 67.41014     |
| std  | 16.488371    | 1185.075926  | 157.9871     |
| min  | 1.000000     | 3.000000     | 0            |
| 25%  | 5.000000     | 250.000000   | 11           |
| 50%  | 7.000000     | 491.000000   | 41           |
| 75%  | 14.000000    | 971.000000   | 102          |
| max  | 1415.000000  | 88376.000000 | 88376        |

 

## **项目结构**

- models
  - textfast.py
  - textcnn.py
  - textrcnn.py
  - textrnn.py
  - transformer.py 

- dataset
  - sougouNews  (数据集)
- public
  - log
    - 日志文件列表（记录训练的数据）
  - path  定义路径 
  - torch_train  模型训练相关
- dataprocess.py  数据处理
- train.py 训练模型相关
- train_all.py  训练所有模型



## **训练结果**

注意：详细训练结果保存在 public/log 文件夹下



**以下结果去验证集中最好的结果（分别对字级别和词级别进行训练）**

| 网络        | 字级别(准确率) | 词级别（准确率） |
| ----------- | -------------- | ---------------- |
| FastText    | 0.9410         | 0.9650           |
| TextCNN     | 0.9662         | 0.9693           |
| TextRNN     | 0.9661         | 0.9702           |
| TextRCNN    | 0.9705         | 0.9712           |
| Transformer | 0.9644         | 0.9684           |

**结果分析**：从训练结果来看，每个网络的准确率都很高，而且差距都很小，应该是样本大部分类别特征很明显，所以导致无论用哪一个网络效果都很好，然而也不能提到非常高的准确率应该是存在一部分样本是很难区分的，甚至是带有迷惑性的。从结果来看，对于特征明显的长文本，此种处理方法是可行的。

