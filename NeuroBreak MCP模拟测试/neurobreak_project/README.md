# NeuroBreak: 大语言模型越狱机制研究

## 项目简介
NeuroBreak是一个基于MCP（Model Context Protocol）工具的大语言模型越狱机制研究项目。该项目旨在揭示大语言模型内部的越狱机制，并分析不同攻击方法的有效性。

## 功能特性
- 🔍 多种越狱攻击方法测试
- 🧠 内部机制深度分析
- 📊 可视化分析工具
- 🗄️ 实验数据管理
- 📈 综合报告生成

## 环境要求
- Python 3.8+
- CUDA支持（可选）
- 相关API密钥（OpenAI、Anthropic、HuggingFace）

## 快速开始

### 1. 环境设置
```bash
python setup_environment.py
```

### 2. 配置API密钥
复制`.env.template`为`.env`并填入您的API密钥：
```bash
cp .env.template .env
# 编辑.env文件，填入您的API密钥
```

### 3. 运行实验
```bash
python run_experiment.py
```

## 项目结构
```
neurobreak_project/
├── src/                    # 源代码
│   ├── jailbreak_attacks.py    # 越狱攻击实现
│   └── mechanism_analyzer.py    # 机制分析器
├── data/                   # 数据目录
├── results/               # 结果目录
├── models/                # 模型目录
├── config.json           # 实验配置
├── requirements.txt      # Python依赖
└── run_experiment.py     # 实验运行脚本
```

## MCP工具配置
项目使用以下MCP工具：
- Python执行器：运行实验代码
- HuggingFace集成：访问开源模型
- OpenAI/Anthropic API：测试商业模型
- 数据库连接：存储实验数据
- 文件系统访问：管理数据文件
- Jupyter支持：交互式分析

## 实验配置
在`config.json`中可以配置：
- 测试模型列表
- 攻击类型
- 测试查询
- 分析深度

## 结果分析
实验完成后会生成：
- 越狱测试结果
- 内部机制分析
- 可视化图表
- 综合研究报告

## 注意事项
- 请确保遵守相关API的使用条款
- 实验数据仅供研究使用
- 请负责任地使用研究成果

## 贡献
欢迎提交Issue和Pull Request来改进项目。

## 许可证
MIT License
