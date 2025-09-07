---
title: 例会环境配置（二）- Conda
author: "KBJiaHeng"
description: 例会环境配置的第二篇教程！为了能顺利参加 AIA 学社的后续例会，让我们一起在电脑上搭建好基本的编程环境吧~
date: 2025-09-06
difficulty: "初级"
readTime: "30min"
tags: ["环境配置"]
---
## 二、Conda 配置

在上一部分中，你已经配置好了 vscode 和 python，我们已经有了两个非常强大的工具！现在我们来安装另一个强大的工具：conda，一种环境管理器。

### 1. Conda 是干啥的

Python 的一大优势是拓展性极强。所谓拓展性，就是说在 python 编写人员提供的基础工具上，众多网友可以利用这些工具构建更强大的工具，然后发布到网上（一般是 github，我们将会在第四次培训的时候介绍它），供你使用。这样被重新构建的更好用的工具，一般被打包成"package"供大家一起使用，这个行为就是对“[开源](https://en.wikipedia.org/wiki/Free_software_movement)”的简单解释。譬如说，我们机器学习中常用的两个 package 就是 Pytorch 和 Tensorflow，前者是 meta 公司发起的项目，后者是 Google。

不难想象，如果我要开发 package A，很有可能我也会站在别人的肩膀上去实现我想要的功能，譬如用 package B ，而不是全部从头开始。此时别人如果要使用 package A，就一定得下载 package B，否则无法正常工作。Package A 和 package B 的关系就叫作依赖关系（dependency）。

打个比方：

- 你写了一份数据分析工具（package A），里面调用了另一个绘图库（package B）的 plot() 函数来画图。
- 你当时写代码时，B 的版本是 2.0，一切正常。
- 结果 B 更新到了 3.0，开发者觉得 plot() 过时了，就直接删掉了。

这时候，如果你电脑上只有 package B v3.0，那么 package A v1.0 就跑不起来了。更尴尬的是：你还想单独用 B v3.0 的一些新功能（比如交互式可视化），但又必须保留 B v2.0 才能让 A 正常工作。这就是一个典型的“依赖冲突”(dependency conflict)问题。

难道鱼和熊掌不可兼得吗？

可以的！Conda 就是帮你解决这种情况的神器。Conda 是一种环境管理器，它允许你为不同的项目创建不同的[环境(environment)](https://www.anaconda.com/docs/tools/working-with-conda/environments)，每个环境里都可以有自己的一套 Python 和依赖包，互不干扰。这样一来，相互冲突的环境可以在一台电脑上共存，我们只需要在使用的时候调用我们需要的环境即可。简直完美！

环境管理器和包管理器有所不同。如果只是想管理 Python 的package，进行下载、更新和删除，我们还可以使用 python 自带的 pip 工具。但是 pip 工具并没有环境的概念，不能处理依赖冲突。所以我们可以在不同的虚拟环境里使用 pip 工具，来实现 python 虚拟环境的包管理。在之后的教程里，我们将采取这种方式。

### 2. 安装 Anaconda

#### **2.1 下载安装包**

Conda 提供了两种方案：

- [Anaconda](https://www.anaconda.com/download) 为你下载好了大家最常用的一些包，譬如用于绘图的 matplotlib，科学计算的 numpy 等。
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) 是 Mini 版本的 Anaconda，只会提供一部分的必要包，占用的空间较小。

点击上面两个链接中的任意一个，根据你的需求选择一个下载吧！

#### **2.2 安装ing**

下载好安装包之后，双击运行。

- 如果你用的是 Windows，一路默认即可；
- 如果你用的是 Linux，在最后一步安装时，会遇到 `yes` 和 `no`。对于初学者，我们建议大家选择 `yes`；如果你刚刚坚定地选择 `no`，或者想要了解他们的区别，请跳转到 **教练，我还要学！** 的 ***如果不把 conda 配置到环境变量**。

#### **2.3 安装成功了吗？**

- 如果你用的是 Windows，请在最底下的搜索栏查找 `Anaconda Prompt` 并打开。
- 如果你用的是 Linux，请打开你的 `terminal`。

在打开的终端输入

```
conda --version
```

如果你看到类似的输出：

```
conda 24.7.1
```

恭喜你，说明你安装成功了！

### 3. [Anaconda 怎么用](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)

这个部分将会带着你创造一个叫 *AIA* 的环境，以便之后我们的教学例会中所使用的包不会和你未来课程中使用的包产生冲突。

#### **3.1 创造 AIA 环境**

刚刚打开的终端想必你还没有关上。请你在终端输入

```
conda create --name AIA python=3.10
```

此时你可能会遇到类似于

```
...
proceed ([y]/n)?
```

的语句，输入 `y`加回车即可。

#### **3.2 激活你的环境**

正如上面所说，conda 是可以管理环境的。这就意味着你可以选择进入或退出某一个自定义的环境。譬如我们要进入*AIA*环境，你可以输入

```
conda activate AIA
```

我们将会在下一次教程中告诉你要下载哪些包，所以此时你不用太在意在这个环境中要做什么，可以输入如下指令退出

```
conda deactivate
```

如果你想要新建别的名字的环境，只需要把上述流程中的 `AIA` 替换为你喜欢的名字。

### 4. 教练，我还要学！

#### **4.1 一些常见的指令**

最好的学习资料就是[官方 documentation](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)。下面我列举几个常用的指令，供大家把玩。

- 如果你想看看自己创建了哪些环境：

  ```
  conda info -e
  ```

  或者

  ```
  conda env list
  ```
- 譬如说你一开始打错 AIA 名字了，你写成了 `ILoveAIA`。如果想改掉这么长的名字，但是你又不想改变你已经配好的环境。那就复制这一个环境，取名叫 `AIA`，然后删除原来的环境。

  ```
  conda create --clone ILoveAIA --name AIA
  conda remove --name ILoveAIA --all
  ```

  （请保持对 AIA 的热爱 qwq）
- 下载 名字叫做 `<PACKAGE>`的 `PACKAGE`

  ```
  conda install <PACKAGE>
  ```
- can can need 这个环境里有什么包

  ```
  conda list
  ```
- 有时候你发现自己下载包的时候下错环境了，没关系，conda 会记录你的修改，你可以通过

  ```
  conda list --revisions
  ```

  得知自己上一次修改被标记为多少。具体来说，你可能得到这样的输出

  ```
  2025-06-13 21:20:58  (rev 0)
  ```

  说明这是第 0 次修改。譬如说你现在做的是第 1 次修改.如果你想要回到第 0 次修改的状态，那就执行

  ```
  conda install --revision 0
  ```

#### **4.2 如果不把 conda 配置到环境变量**

首先，请自行了解什么是环境变量。简单来说，如果把 conda 配置到环境变量，那么终端可以在任何地方直接使用 conda 指令。

Windows 下安装时，如果一路默认并不会把 conda 配置到环境变量中。所以在 PowerShell 或者 cmd 终端中输入 `conda`，终端并不能成功识别；但 Windows 下我们有 Anaconda Prompt，在这个终端里我们可以自由使用 conda 指令。

Linux 下安装时：

- 如果选择 `yes`，你将把 conda 配置到你的环境变量中；这可能会方便你使用 conda，但你在每一次开启终端的时候都会有大约一秒的延迟，深度使用终端的人会明显感受到差异。
- 如果选择 `no`，由于没有将 conda 配置到环境变量中，你将无法在终端里直接使用 `conda` 指令，而是需要找到 conda 可执行文件，并在使用的时候指明完整路径。如此一来，终端才能正确执行。例如：

  - 想要查看 conda 版本，需要在 conda 的目录下找到可执行文件 `conda`：
    - 终端输入：

      ```
      /路径/Anaconda3/bin/conda --version
      ```
      其中 `路径` 替换为你的 `Anaconda3` 的存储路径；如果你安装的是 Miniconda，还需要把 `Anaconda3` 替换为 `Minconda`。
  - 想要激活虚拟环境，需要在 conda 的目录下找到可执行文件 `activate`：
    - 终端输入：

      ```
      /路径/Anaconda3/bin/activate 虚拟环境名
      ```
  - 其他操作同理

### 附录

- `2025.9.7` 感谢李奕辰同学指出 `conda deactivate` 处原有的 typo