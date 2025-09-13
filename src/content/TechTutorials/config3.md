---
title: 例会环境配置（三）- Git工具使用和环境检测
author: "曾睿鸣"
description: 例会环境配置的第三篇教程！为了能顺利参加 AIA 学社的后续例会，让我们一起在电脑上搭建好基本的编程环境吧~
date: 2025-09-13
difficulty: "初级"
readTime: "30min"
tags: ["环境配置"]
---

在例会环境配置的第三部分，我们将学会使用 **git 工具** 从 GitHub 上下载（clone）社团的代码仓库，并通过仓库里的检测文件确认自己的环境已经配置好。

## 一、GitHub 是什么？

GitHub 是目前全球 **最大的开源社区** 和 **代码托管平台**。  
世界上大多数知名开源项目（如 Linux、PyTorch、TensorFlow）都托管在 GitHub 上；任何人都能下载、学习这些代码，还可以贡献自己的修改。  

### 我们可以用 Github 做什么？

1. **学习**：直接阅读和使用世界一流的开源代码。  
2. **协作**：多人开发同一个项目，保持版本一致。  
3. **展示**：程序员常把自己的作品放在 GitHub 上。  

### 在我们社团的使用场景

- 我们的活动资料、示例代码都会放在 GitHub 仓库里。  
- 只要用一条命令，就能把仓库下载到本地；如果有更新，也能随时同步。  

## 二、Git 工具是什么？为什么要用它？

Git 是一个 **版本控制工具**，最早是 Linux 之父 Linus Torvalds 发明的。  
听起来很高大上，其实你只要知道它帮我们解决了三个问题：

1. 管理代码的“历史”：Git 可以记录代码的修改历史，就像 Word 文档里的“撤销/恢复”功能，但更强大。比如你写了一个神经网络，改了 10 次，Git 可以帮你随时回到第 5 次的版本。  
2. 团队协作：多个人写同一个项目时，如果都自己复制粘贴，很容易版本混乱；使用 Git，大家可以同时开发，最后再把修改合并起来。  
3. 下载和同步 GitHub 仓库：Git 和 GitHub 是“工具 + 云端”的关系。GitHub 上存放代码，Git 工具帮你把代码拉取（clone）到本地。以后社团更新了仓库，你只需要一条命令就能同步到最新版本。  

## 三、下载 Git 工具

1. 打开 [Git 官方网站](https://git-scm.com/download/) 下载 Git 安装包。  
2. 按照提示安装，一路点击“Next”即可。  
3. 安装完成后，打开 **命令提示符 (cmd)** 或 **PowerShell**，输入：
   ```bash
   git --version
   ```

✅ **Checkpoint**"：如果显示版本号（例如 `git version 2.x.x`），说明安装成功。

## 四、从社团仓库 clone 检测环境仓库

检测环境仓库 [地址](https://github.com/NJU-AIA/TestEnv)。

### 1. 在你想要 clone 仓库的位置打开终端（PowerShell / Git Bash）。

- 终端每一行最前面会有一长串，代表终端当前所处的位置，也就是当前目录；如果没有特别在指令中指出，指令的默认执行位置都是当前的目录。
- `git clone` 指令会把相关仓库中的内容 clone 到当前目录下。
- 因此，在 `git clone` 之前，我们应该先进入我们想要 clone 的目录。方法有两种：
  - 直接在那个目录打开终端（可以直接用 VSCode 打开目录，然后在左上方调出终端）；
  - 打开终端之后，使用 `cd` 指令进入想要的文件夹。
  
### 2.输入以下命令，把仓库下载到本地：

   ```bash
   git clone https://github.com/NJU-AIA/TestEnv.git
   ```

✅ **Checkpoint**：在 `TestEnv` 文件夹里里应该能看到：

```
testenv.py
testenv.ipynb
README.md
```

---

## 五、利用检测文件确认环境

在 VSCode 中打开 `TestEnv` 文件夹，我们要来运行下面两个文件：

### 1. 运行 `testenv.py`

最下方搜索栏输入 `Anaconda Prompt`，找到并打开 Anaconda Prompt。

激活 `AIA` 虚拟环境。

进入 `TestEnv` 目录（在 clone 的时候，我们应该已经学会如何进入我们想要的目录），之后输入：

```bash
python testenv.py
```

运行结果会：

* 检查 numpy / matplotlib 等库是否安装成功。
* 自动生成一张测试图 `matplotlib_test.png`。

如果终端输出结果如下所示：

```bash
===== 环境检测开始 =====

=== 安装检测 ===
[OK] struct 已安装
[OK] os 已安装
[OK] urllib.request 已安装
[OK] zipfile 已安装
[X] numpy 未安装，请先运行: pip install numpy
[X] matplotlib 未安装，请先运行: pip install matplotlib

=== 测试 numpy ===
[X] numpy 测试失败: No module named 'numpy'

=== 测试 matplotlib ===
[X] matplotlib 测试失败: No module named 'matplotlib'

===== 环境检测结束 =====
```

则需要根据提示输入指令安装相应的包。

✅ **Checkpoint**：命令行输出如下信息：

```bash
===== 环境检测开始 =====

=== 安装检测 ===
[OK] struct 已安装
[OK] os 已安装
[OK] urllib.request 已安装
[OK] zipfile 已安装
[OK] numpy 已安装
[OK] matplotlib 已安装

=== 测试 numpy ===

Numpy 测试成功: [1 2 3] 平方: [1 4 9]

=== 测试 matplotlib ===

===== 环境检测结束 =====
```

并且目录下出现 `matplotlib_test.png`。


### 2. 运行 `testenv.ipynb`

首先在虚拟环境里安装 **Jupyter** 依赖：  

   ```bash
   pip install notebook jupyter
   ```


之后在 VSCode 中搜索并安装 `jupyter` 扩展。

打开 `testenvjpy.ipynb`，文件会以 **Notebook 形式**显示，每段代码是一个 **cell**。  

Jupyter Notebook 需要选择一个 kernel 来解释运行代码，点击右上角的 `select kernel` ，之后找到并选择 `AIA` 虚拟环境里的 python 解释器。

第一次运行的时候 VSCode 可能会要求下载一些辅助的包，有弹窗时点击确认进行安装即可。

从上到下逐个点击 cell 左侧的 ▶（运行按钮），即可按顺序执行代码；如果想省事，也可以直接点击最上方的 `Run All`。  

✅ **Checkpoint**： 在 `testenvjpy.ipynb` 的每一个 cell 下方可以看到这个块的运行结果。第一个块下方是 numpy 和 matplotlib 的版本信息，第二个块下方是一条 S 型的 sigmoid 函数曲线。

## 六、教练！我还想学

关于 Git 工具的使用，可以参看 [廖雪峰的官方网站](https://liaoxuefeng.com/books/git/introduction/index.html)。

git clone 时，使用 https 速度比较慢，并且不总是可以成功。相比之下，使用 ssh 是更好的选择。如果感兴趣的话，可以注册一个 github 帐号，并参照 [官方文档](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) 配置 SSH key 并 clone 一些你喜欢的代码仓库。