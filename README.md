# Paper Summary Agent Lab

一个面向求职展示的 Agent 学习项目。
当前阶段先实现“论文摘要结构化总结工具”，后续会逐步扩展为 Tool Calling、RAG、Agent Loop、论文阅读 Agent 和求职辅助 Agent。

## 1. 项目目标

本项目的目标是从一个最小可运行的大模型应用开始，逐步构建一个可展示、可解释、可扩展的 Agent 项目。

当前已完成的能力：

* 调用 OpenAI-compatible LLM API
* 使用 `.env` 管理 API Key 和模型配置
* 读取论文摘要文本
* 使用 LLM 生成结构化 JSON 总结
* 使用 Pydantic 校验输出格式
* 支持单文件处理
* 支持批量处理 `data/papers` 目录下的多个 `.txt` 文件
* 保存运行日志，方便调试 prompt、模型输出和错误信息

当前项目还不是完整 Agent，而是 Agent 项目的基础模块：
LLM 结构化摘要工具 / Agent 原型地基。

后续会继续加入：

* Tool Calling
* Agent Loop
* RAG 检索
* Memory
* Workflow
* Evaluation
* Web Demo

## 2. 项目结构

```text
agent/
│  .env
│  .env.example
│  requirements.txt
│  README.md
│
├─app
│      llm.py
│      main.py
│      schemas.py
│
├─data
│  └─papers
│          demo.txt
│          agent_test.txt
│
└─outputs
    │  demo_summary.json
    │  agent_test_summary.json
    │
    └─logs
        prompt、raw_output、error 等日志文件
```

## 3. 环境准备

本项目使用 Conda 环境：

```cmd
conda activate agent
```

进入项目目录：

```cmd
cd /d D:\codex_projects\projects\agent
```

如果 cmd 中文显示乱码，可以先执行：

```cmd
chcp 65001
```

安装依赖：

```cmd
pip install -r requirements.txt
```

## 4. 环境变量配置

项目使用 `.env` 文件保存 API 配置。

`.env.example` 示例：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

实际使用时，新建 `.env` 文件，并填入自己的真实 API Key。

注意：

* 不要把真实 `.env` 提交到 GitHub
* 真实 API Key 只保存在本地
* GitHub 只提交 `.env.example`

## 5. 单文件运行

默认处理：

```cmd
python app\main.py
```

等价于处理：

```text
data\papers\demo.txt
```

指定输入文件：

```cmd
python app\main.py --input data\papers\demo.txt
```

或：

```cmd
python app\main.py --input data\papers\agent_test.txt
```

指定输出文件：

```cmd
python app\main.py --input data\papers\demo.txt --output outputs\test_result.json
```

运行成功后，会看到类似输出：

```text
已生成结构化总结：
D:\codex_projects\projects\agent\outputs\demo_summary.json
```

## 6. 批量运行

批量处理 `data\papers` 目录下所有 `.txt` 文件：

```cmd
python app\main.py --batch
```

运行成功后，会看到类似输出：

```text
开始批量处理，共发现 2 个 txt 文件。
[成功] agent_test.txt -> agent_test_summary.json
[成功] demo.txt -> demo_summary.json

批量处理完成。
成功：2
失败：0
```

也可以指定输入目录和输出目录：

```cmd
python app\main.py --batch --input-dir data\papers --output-dir outputs
```

## 7. 输入格式

输入文件是普通 `.txt` 文件，内容为一段论文摘要。

示例：

```text
Large language model agents often suffer from unreliable tool selection,
hallucinated intermediate reasoning, and lack of systematic evaluation...
```

文件路径示例：

```text
data\papers\demo.txt
```

## 8. 输出格式

输出是结构化 JSON 文件。

示例：

```json
{
  "problem": "现有的大语言模型代理系统存在不可靠的工具选择、幻觉的中间推理以及缺乏系统评估的问题。",
  "method": "本文提出了一个模块化代理框架，该框架将任务规划、工具执行、内存管理和验证分离开来。",
  "datasets": [
    "网络导航基准",
    "知识密集型问答基准"
  ],
  "contributions": [
    "提高了任务成功率",
    "减少了工具使用错误"
  ],
  "limitations": []
}
```

字段说明：

| 字段            | 含义        |
| ------------- | --------- |
| problem       | 论文要解决的问题  |
| method        | 论文方法概述    |
| datasets      | 摘要中提到的数据集 |
| contributions | 主要贡献      |
| limitations   | 摘要中提到的局限  |

## 9. 日志说明

程序会把运行过程中的 prompt、模型原始输出和错误信息保存到：

```text
outputs\logs
```

日志主要用于：

* 检查 prompt 是否合理
* 检查模型原始输出是否符合 JSON
* 定位 JSON 解析错误
* 定位 Pydantic 校验错误
* 定位 API 调用失败原因

## 10. 当前功能边界

当前项目已经支持：

* 单文件论文摘要结构化总结
* 批量论文摘要结构化总结
* JSON 解析
* Pydantic 校验
* 命令行参数
* 运行日志
* API 调用失败重试

当前项目暂不支持：

* PDF 直接解析
* 多轮 Agent Loop
* 工具调用
* RAG 检索
* 向量数据库
* 论文全文问答
* Web 页面展示

## 11. 当前项目定位

当前阶段不是完整 Agent，而是 Agent 项目的第一层基础模块。

可以理解为：

```text
LLM API + Structured Output + CLI + Logging
```

这是后续构建 Agent 的地基。后续会在此基础上继续加入工具调用、检索增强、工作流和评估模块。

## 12. 后续计划

下一阶段计划：

1. 增加更稳定的 JSON 解析失败重试机制
2. 增加 README 示例截图
3. 增加 `.gitignore`
4. 实现 calculator 工具
5. 实现 read_file 工具
6. 实现 search_paper 工具
7. 构建最小 Tool Calling Agent
8. 扩展为论文阅读 RAG Agent

## 13. 示例命令汇总

```cmd
cd /d D:\codex_projects\projects\agent
conda activate agent
chcp 65001
python app\main.py
python app\main.py --input data\papers\demo.txt
python app\main.py --input data\papers\agent_test.txt
python app\main.py --input data\papers\demo.txt --output outputs\test_result.json
python app\main.py --batch
python app\main.py --batch --input-dir data\papers --output-dir outputs
```
