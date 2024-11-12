---
title: qqbot部署
type: learning
description: 一个稳定部署qqbot的方案
abbrlink: 3114420454
---

## 方案介绍

想要部署一个 QQ 机器人，首先要有一个聊天机器人的框架用来处理消息，这里选择  (nonebot)[https://nonebot.dev/docs]  。其次需要让机器人对接到 QQ 账号实现消息的收发，目前来看最稳定的方案是使用新版QQ的插件[LLOneBot](https://llonebot.github.io/zh-CN/)。 

> Caution！
>
> 请不要在 QQ 官方群聊和任何影响力较大的简中互联网平台（包括但不限于: 哔哩哔哩，微博，知乎，抖音等）发布和讨论任何与本插件存在相关性的信息

## 快速部署 nonebot

有能力的同学建议直接观看官方文档，接下来的内容是写给没有配置环境基础的同学的。

### python 环境准备

> - 请确保你的 Python 版本 >= 3.9
> - **我们强烈建议使用虚拟环境进行开发**，如果没有使用虚拟环境，请确保已经卸载可能存在的 NoneBot v1！！！

建议使用 conda/miniconda 创建虚拟环境

去 [conda官网](https://docs.conda.io/projects/conda/en/stable/) 下载安装包一件安装即可，C 盘不够大的同学记得修改安装路径，因为 conda 环境多起来的话会占用相当多硬盘空间。 

安装完 conda 你会发现在终端不能直接使用conda命令，如果你想要通过直接在命令行中键入`conda activate`命令打开任意conda环境。可以参考这篇文章： [安装Anaconda（miniconda）后如何在powershell使用conda activate命令（Windows）-CSDN博客](https://blog.csdn.net/m0_57170739/article/details/134833229) 

安装完成后创建并进入虚拟环境

```bash
conda create --name QQbot
conda activate QQbot
```

因为nonebot 官方文档十分详细（甚至有可以直接复制文本的视频），接下来直接按照 [快速上手 | NoneBot](https://nonebot.dev/docs/quick-start) 操作即可。如果你没有看懂，或者觉得我的教程比官方文档更可靠，可以看下面我的操作。

{% asciinema /record/example.cast true true 2 %}

###  [安装脚手架](https://nonebot.dev/docs/quick-start#安装脚手架)   

确保你已经安装了 Python 3.9 及以上版本，然后在命令行中执行以下命令：

1. 安装 [pipx](https://pypa.github.io/pipx/)

   ```bash
   python -m pip install --user pipx
   python -m pipx ensurepath
   ```

   

   如果在此步骤的输出中出现了“open a new terminal”或者“re-login”字样，那么请关闭当前终端并重新打开一个新的终端。

2. 安装脚手架

   ```bash
   pipx install nb-cli
   ```

安装完成后，你可以在命令行使用 `nb` 命令来使用脚手架。如果出现无法找到命令的情况（例如出现“Command not found”字样），请参考 [pipx 文档](https://pypa.github.io/pipx/) 检查你的环境变量。

### 创建运行项目



当你看到如下内容，表示 nonebot 成功运行：

```powershell
(QQbot) PS F:\QQbot\test> nb run
使用 Python: F:\QQbot\test\.venv\Scripts\python.exe
10-06 12:28:17 [SUCCESS] nonebot | NoneBot is initializing...
10-06 12:28:17 [INFO] nonebot | Current Env: dev
10-06 12:28:17 [DEBUG] nonebot | Loaded Config: {'driver': '~fastapi', 'host': IPv4Address('127.0.0.1'), 'port': 8080, 'log_level': 'DEBUG', 'api_timeout': 30.0, 'superusers': set(), 'nickname': set(), 'command_start': {'/'}, 'command_sep': {'.'}, 'session_expire_timeout': datetime.timedelta(seconds=120), 'environment': 'dev'}
10-06 12:28:18 [DEBUG] nonebot | Succeeded to load adapter "OneBot V11"
10-06 12:28:18 [SUCCESS] nonebot | Succeeded to load plugin "echo" from "nonebot.plugins.echo"
10-06 12:28:18 [SUCCESS] nonebot | Running NoneBot...
10-06 12:28:18 [DEBUG] nonebot | Loaded adapters: OneBot V11
10-06 12:28:18 [INFO] uvicorn | Started server process [17284]
10-06 12:28:18 [INFO] uvicorn | Waiting for application startup.
10-06 12:28:18 [INFO] uvicorn | Application startup complete.
10-06 12:28:18 [INFO] uvicorn | Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```

可以看到 nonebot 在本地的 8080 端口使用开启了服务，使用 OneBot V11 协议。

## 利用LLOneBot 插件开启 websocket 反向代理

### 安装插件

**首先检查自己的QQ是否是NTQQ，旧版本的 QQ 和 Tim 无法安装插件。**  

安装教程：[快速开始 | LLOneBot](https://llonebot.github.io/zh-CN/guide/getting-started)  

因为QQ会检查文件完整性，安装插件失败可能导致无法打开QQ，可以重装QQ尝试别的方案。

（我使用通用安装方法：使用 [一键安装脚本](https://github.com/Mzdyl/LiteLoaderQQNT_Install/releases) 安装 LiteLoaderQQNT，然后下载 [LLOneBot](https://github.com/LLOneBot/LLOneBot/releases) 最新版本解压放到 `plugins` 目录下，然后重启 QQ ）

### 对接

正确安装插件之后在设置中可以看到 LLOneBot 的配置

![](/images/llonebot.png)

只用开启反向 WebSocket 监听服务，地址填写 `ws://127.0.0.1:8080/onebot/v11/ws ` 

这时候再运行 nonebot 

```powershell
PS F:\QQbot\test> nb run
使用 Python: F:\QQbot\test\.venv\Scripts\python.exe
10-06 13:17:16 [SUCCESS] nonebot | NoneBot is initializing...
10-06 13:17:16 [INFO] nonebot | Current Env: dev
10-06 13:17:16 [DEBUG] nonebot | Loaded Config: {'driver': '~fastapi', 'host': IPv4Address('127.0.0.1'), 'port': 8080, 'log_level': 'DEBUG', 'api_timeout': 30.0, 'superusers': set(), 'nickname': set(), 'command_start': {'/'}, 'command_sep': {'.'}, 'session_expire_timeout': datetime.timedelta(seconds=120), 'environment': 'dev'}
10-06 13:17:16 [DEBUG] nonebot | Succeeded to load adapter "OneBot V11"
10-06 13:17:16 [SUCCESS] nonebot | Succeeded to load plugin "echo" from "nonebot.plugins.echo"
10-06 13:17:16 [SUCCESS] nonebot | Running NoneBot...
10-06 13:17:16 [DEBUG] nonebot | Loaded adapters: OneBot V11
10-06 13:17:17 [INFO] uvicorn | Started server process [4300]
10-06 13:17:17 [INFO] uvicorn | Waiting for application startup.
10-06 13:17:17 [INFO] uvicorn | Application startup complete.
10-06 13:17:17 [INFO] uvicorn | Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
10-06 13:17:17 [INFO] uvicorn | ('127.0.0.1', 64214) - "WebSocket /onebot/v11/ws" [accepted]
10-06 13:17:17 [INFO] nonebot | OneBot V11 | Bot 1561365020 connected
10-06 13:17:17 [INFO] websockets | connection open
```

你可以发现成功连上了QQ 

如果有人给你发消息，消息的信息会在控制台显示，

```powershell
10-06 13:19:14 [SUCCESS] nonebot | OneBot V11 1561365020 | [message.private.friend]: Message 526119788 from 3792761726 '你好'
10-06 13:19:14 [DEBUG] nonebot | Checking for matchers in priority 1...
10-06 13:19:14 [DEBUG] nonebot | Checking for matchers completed
10-06 13:19:38 [SUCCESS] nonebot | OneBot V11 1561365020 | [message.private.friend]: 
```

如果你安装了 echo 插件（默认选项会安装）那么别人给你发送消息 "/echo hello" 的时候，none 会进行相应并且进行回声 (给别人发送 "hello")

![](/images/echo.png)

