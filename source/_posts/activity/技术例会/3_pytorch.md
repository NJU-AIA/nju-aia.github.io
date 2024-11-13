---
abbrlink: 2749833372
title: 3月技术分享例会：pytorch速通教学
type: activity
description: 关于numpy，pytorch等常用组件内容
date: 2024-05-11
author: 南的AIA
num: 3
---

2024年3月31日，南大AIA在南大仙林校区开展了3月技术分享例会。本次例会主题是“pytorch速通教学”，旨在帮助初学或感兴趣的朋友了解numpy，pytorch常用组件内容。

## 环境配置

推荐使用conda进行虚拟环境的配置以保证各个python应用环境依赖库互不冲突。

演示配置：

- Ubuntu 22.04
- python 3.11
- cuda 12.1
- pytorch 2.2.2

安装conda并配置运行环境：（下述指令建议前往mirror.nju.edu.cn里搜索anaconda，miniconda来找到最新符合机器架构的安装脚本）

```shell
wget https://mirror.nju.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

随后按说明配置conda安装位置，是否每次登录都自动激活conda等

随后创建一个新环境并进入其中：(常见问题[1])

注意，下述第3条命令建议前往pytorch官网查看适合您机器配置的命令

```shell
conda create --name torch_test python=3.11 
conda activate torch_test
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```



> [1]：实际例会中，有不少朋友被提示“找不到conda命令”。这种情况发生时：如果安装正常完成的话，可以借助find命令在安装位置找到conda可执行文件，并通过export命令（本终端会话），或修改.bashrc（修改后记得sourse，永久地）（如果是别的sh，请更改对应的配置文件，如.zshrc）将conda添加至PATH变量；如果conda安装未正常完成，可以上网搜搜问题，或来群里问问大伙。

## numpy常用介绍

可以去runoob NumPy 教程学习更为全面的numpy介绍。

这里主要强调

- 数组创建
- 数据类型
- 数组形状
- 切片索引访问

特别值得注意的是，array默认数据类型int和float64的区别（只有后者可以求导），numpy各维度数组的形状规则，形状改变操作

如果对python索引/切片不太熟悉的话，可以通过“NumPy 切片和索引”，”NumPy 高级索引“章节学习这些神奇方便的功能

## pytorch速通介绍

我们借鉴了pytorch官网教程以及pytorch 60min blitz，深入浅出pytorch教程，同时也参考一篇知乎的60min blitz节选翻译.

更完整的各功能文档教学可查看PyTorch recipes但是中文，或者更全的中文pytorch资源.

主要讲解以下各点：

- tensor张量：其实就是pytorch版的numpy数组（可扔GPU里算）
- 运算操作（尤其是利用广播机制）
- 自动求导autoGrad（一维函数求导/多维下求Jacobi矩阵得数值）与梯度累计操作
- 预定义的连接层，卷积池化层
- 预定义的损失函数，优化器
- 神经网络结构与反向传播（BP）原理
- 神经网络速搭与代码解释
- 转移数据至GPU与显卡设备
- 多卡训练与数据平行的模型转化

#### 神经网络速搭

1  **定义网络结构**：(定义组件，链接组件)

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
```

2  **准备优化器**

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

3  **开始训练的代码**

```
import time
start = time.time()
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入数据
        inputs, labels = data
        # 清空梯度缓存
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:
            # 每 2000 次迭代打印一次信息
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training! Total cost time: ', time.time()-start)
```

训练完成后，可以检查模型准确性[2]：

```
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
```



> [2]：会上其实也提到过，此处用训练集数据做测试是极不可取的行为！测试集(test)应严格与训练集(train)分开，不可用同一批数据。



[阅读原文](https://mp.weixin.qq.com/s/sjMAsNXcpL3vUmH6tqTIYg)
