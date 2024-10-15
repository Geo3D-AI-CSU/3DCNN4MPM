# 基于卷积神经网络的三维成矿远景区预测  
随着地表和浅层矿产资源的不断开采，全球对隐矿床勘探的需求不断增加。然而，隐藏矿物前景建模 （MPM） 需要付出巨大的努力，尤其是在三维建模的方法和技术方面。在这项研究中，我们提出了一种用于MPM的轻量级三维卷积神经网络（3D CNN），它采用了GoogleNet的初期结构，并结合了端到端学习的思想。
模型基于Tensorflow1.12框架，设计了轻量级卷积神经网络用于三维成矿远景区预测，相较于传统的卷积神经网络，其轻量级主要体现在通过3 $\times$ 3 $\times$ 3的小卷积核替代5 $\times$ 5 $\times$ 5和7 $\times$ 7 $\times$ 7的大卷积核，引入GoogelNet的inceptionV3结构、使用反卷积和特征降维模块替代3个全连接层以降低模型的参数量，达到减少计算复杂度的目的。以下为实验用到的数据、代码及环境说明。  
论文链接：<https://www.sciencedirect.com/science/article/pii/S0169136823005048#t0020>  

***
# 目录  
- [文件介绍](#文件介绍)  
- [安装](#安装)  
- [鸣谢](#鸣谢)  
  
# 文件介绍  
文件列表包括以下内容：
- README
- light_CNN_New_Data
- light_CNN
- fc_CNN
- installment
- dataset  
 
每个文件内容基本由训练过程的代码、模型定义及函数合集两部分组成。

**README**：项目及文件介绍

**light_CNN_New_Data**：这是最初论文即论文里所提到的轻量级卷积网络模型，它与light-CNN文件的区别在于，该文件中的训练过程的代码经过了优化，具体表现在：1.改进了数据读取和模型训练的过程。并行地一次性读取训练数据到内存，然后再开始训练，避免训练过程频繁的IO操作，缩短了训练时间，原来模型一个epoch需要2分钟的时间，优化后一个epoch只需要12秒。2.数据来源不同，训练数据为原来CSV表格汇总成的numpy数据，可以通过设置参数自由调整训练的正负样本比例。<u>fc_CNN_train是训练代码，model是自定义的模型和函数集合</u>

**light_CNN**:这是最初版本即论文里所提到的轻量级卷积网络模型，未做任何改动的代码<u>。其中evaluate_nockpt是模型预测并将预测结果输出到SQL Server的代码。light_CNN_train则是训练代码，model是自定义的模型和函数集合，

**fc_CNN**：使用5 $\times$ 5 $\times$ 5、7 $\times$ 7 $\times$ 7大卷积核、3个全连接层以及重采样数据的模型。<u>其中，fc_CNN_train是训练代码，model是自定义的模型和函数集合，train_data是原CSV数据重新导出的以npy格式存储的numpy数组。</u>

**installment**:训练代码所使用的环境依赖信息，方便环境配置，以requirement.txt和environment.yml不同格式的文件保存。  

**dataset**:
-train_data4fcCNN:allData.npy是以numpy数组格式存储的训练数据，patchesNoOre则是代表不含矿的标签数据,patchesOre则是含矿的标签数据。
-train_data4lightCNN:以CSV表格存储的训练数据，文件夹内包括csv表格，表格表格代表一个采样的patch（每个patch大小为16 $\times$ 16 $\times$ 16，由4096个体素所组成，各列的含义分别是每个体素的I、J、K坐标，7个归一化的控矿因子，倒数最后一列ore1是含矿标签，1代表含矿，0代表不含矿，而倒数第二列ore0则是0代表含矿，1代表不含矿，训练只读入最后一列ore1，目的是是使得训练标签是稀疏矩阵），除csv表格以外，还有test.txt和train.txt文件，分别是存储着csv的文件名，实际使用是通过读取txt的文件名索引进而读取每一个csv文件以完成五折交叉验证。
-train_data4lightCNN_new_data:allData.npy是以numpy数组格式存储的训练数据，patchesNoOre则是代表不含矿的标签数据,patchesOre则是含矿的标签数据。



# 安装  
1. 使用environment.yml创建环境

```
conda env create -f environment.yml
```
解读：在base环境下执行上述指令，会直接创建一个新的环境，并在该环境下，安装相应依赖项
2. 使用requirements.txt创建环境  
```
pip install -r requirements.txt
```
 解读：在当前环境下安装相应依赖项，如果需要在其他环境下安装依赖项，可以先创建并激活新环境，再使用上述命令。 
  
# 鸣谢  
本研究得到了国家自然科学基金（批准号：42072326）和中国地质调查局工作项目（批准号：DD20190156）的资助。作者感谢张春鹏博士、刘宝山博士（中国地质调查局沈阳中心）和刘文玉博士（华东理工大学）在数据收集方面提供的帮助。我们还要感谢中国地理信息系统国家工程研究中心与中南大学共建的 MapGIS 实验室提供的 MapGIS® 软件（武汉中地数码科技有限公司，中国武汉）。
