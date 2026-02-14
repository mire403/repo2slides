<div align="center">

# Repo2Slides - 代码仓库一键变 PPT

</div>

<div align="center">

![Repo2Slides Logo](https://img.shields.io/badge/Repo2Slides-代码一键做PPT-blue?style=for-the-badge&logo=github)

**Turn a GitHub repo into slides. 让代码仓库秒变学术汇报 PPT！**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)](https://openai.com/)

[🚀 快速开始](#-快速开始) • [📖 功能详解](#-核心功能深度解析) • [💡 使用示例](#-使用示例) • [🏗️ 架构设计](#️-架构设计)

</div>

---

## 📌 项目简介

**Repo2Slides** 是一个专为**学生、研究者、开发者**设计的智能工具，能够**自动分析 GitHub 仓库**并生成**结构清晰的学术风格 PPT 大纲**（Markdown slides）。

### 🎓 解决的核心问题

你是否遇到过这些场景？

- 📚 **课程汇报**：需要展示项目，但没时间通读整个 repo
- 🔬 **组会汇报**：导师要求讲代码，但不知道从哪开始
- 🎓 **毕业答辩**：项目代码很多，PPT 不知道如何组织
- 💼 **项目展示**：想快速生成一个能直接用来讲的 PPT

**Repo2Slides 帮你解决这些问题！** 🎉

### ✨ 核心价值

- ⚡ **一键生成**：一行命令，自动分析仓库并生成 PPT 大纲
- 🧠 **智能理解**：自动识别项目结构、方法、实验、结果
- 📊 **深度分析**：支持 ML 框架检测、metrics 文件解析、代码入口点推断
- 🎨 **学术风格**：生成符合学术汇报规范的 8-12 页 PPT
- 🔧 **灵活配置**：支持 LLM 增强模式或纯启发式离线模式

---

## 🚀 快速开始

### 📦 安装

```bash
# 克隆仓库
git clone https://github.com/yourname/Repo2Slides.git
cd Repo2Slides

# 安装依赖
pip install -r requirements.txt
```

**推荐 Python 3.9+** 🐍

### 🎯 基本使用

```bash
# 最简单的用法：分析一个本地仓库
python cli.py ./path/to/your/repo --out slides.md

# 或者使用模块方式
python -m repo2slides ./path/to/your/repo --out slides.md
```

**就这么简单！** ✨ 生成的 `slides.md` 可以直接用 [Marp](https://marp.app/) 或 [Reveal.js](https://revealjs.com/) 打开预览。

---

## 📖 核心功能深度解析

### 1️⃣ Repo Analyzer（仓库分析器）🔍

**功能**：智能扫描仓库结构，提取关键信息

#### 🎯 核心能力

- **目录结构识别**：自动识别 `src/`、`models/`、`experiments/`、`scripts/`、`results/` 等关键目录
- **文件类型检测**：识别 Python、Notebook、配置文件、结果文件等
- **代码解析**：使用 AST 解析 Python 文件，提取类名、函数名、docstring（**不读实现代码**，保证速度）
- **ML 框架检测**：自动识别 PyTorch、TensorFlow、Keras、Scikit-learn、XGBoost、LightGBM、JAX、HuggingFace 等
- **入口点推断**：从文件名和函数名推断 `train`、`eval`、`main` 等入口点

#### 💻 代码示例

```python
from repo2slides.analyzer import RepoAnalyzer
from pathlib import Path

# 初始化分析器
analyzer = RepoAnalyzer()

# 分析仓库
repo_summary = analyzer.analyze(Path("./your_repo"))

# 查看分析结果
print(f"检测到的 ML 框架: {repo_summary.ml_frameworks}")
# 输出: ['PyTorch', 'HuggingFace']

print(f"入口点: {repo_summary.entrypoints[:3]}")
# 输出: ['train.py: function train()', 'eval.py (file)', 'scripts/run_experiment.py (experiment/script)']

print(f"Notebooks: {repo_summary.notebook_files}")
# 输出: ['notebooks/analysis.ipynb', 'notebooks/visualization.ipynb']

print(f"结果文件: {repo_summary.results_files}")
# 输出: ['results/metrics.csv', 'logs/training.json']
```

#### 🔬 深度解析：ML 框架检测原理

```python
# analyzer.py 中的框架检测逻辑
@staticmethod
def _detect_ml_frameworks(path: Path) -> set[str]:
    """
    通过扫描 import 语句检测 ML 框架
    只做轻量级文本匹配，不执行代码
    """
    frameworks: set[str] = set()
    src = safe_read_text(path)
    src_lower = src.lower()
    
    # PyTorch 检测
    if any(imp in src_lower for imp in ("import torch", "from torch")):
        frameworks.add("PyTorch")
    
    # TensorFlow 检测
    if any(imp in src_lower for imp in ("import tensorflow", "from tensorflow")):
        frameworks.add("TensorFlow")
    
    # ... 其他框架类似
    
    return frameworks
```

**设计亮点**：
- ✅ **轻量级**：只扫描 import 语句，不执行代码
- ✅ **快速**：O(n) 时间复杂度，n 为文件数量
- ✅ **准确**：基于常见 import 模式，误报率低

---

### 2️⃣ Content Extractor（内容抽取器）📝

**功能**：从 README 和代码结构中提取结构化信息

#### 🎯 核心能力

- **README 解析**：智能识别中英文章节（背景、方法、实验、结果等）
- **代码结构补充**：当 README 不完整时，从代码结构推断缺失信息
- **Metrics 文件分析**：自动解析 CSV/JSON 结果文件，提取关键指标
- **智能推断**：结合目录结构、入口点、Notebooks 等信息补全内容

#### 💻 代码示例

```python
from repo2slides.extractor import ContentExtractor
from repo2slides.utils import LLMClient, LLMConfig

# 方式1：使用 LLM 增强（推荐，效果更好）
config = LLMConfig(model="gpt-4o-mini", api_key="your_key")
llm_client = LLMClient(config)
extractor = ContentExtractor(llm_client=llm_client)

# 方式2：纯启发式模式（离线，无需 API）
extractor = ContentExtractor(llm_client=None)

# 提取结构化内容
structured_content = extractor.extract(repo_summary)

print(structured_content)
# 输出:
# {
#     "title": "My Awesome ML Project",
#     "background": "This project aims to...",
#     "method": "We propose a novel approach using...",
#     "architecture": "The system consists of...",
#     "experiments": "We conducted experiments on...",
#     "results": "Our method achieves 93.2% accuracy...",
#     ...
# }
```

#### 🔬 深度解析：Metrics 文件智能分析

```python
# extractor.py 中的 metrics 分析逻辑
@staticmethod
def _analyze_csv_metrics(path: Path, rel: str) -> str:
    """
    分析 CSV metrics 文件，提取最佳值和最后值
    支持常见指标：accuracy, f1, loss, val_loss 等
    """
    import csv
    
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # 识别指标列
        metric_keywords = {
            "acc": "accuracy",
            "f1": "f1",
            "loss": "loss",
            ...
        }
        
        # 提取数值并计算最佳/最后值
        for col, label in metric_cols.items():
            values = [float(row[col]) for row in rows]
            best_val = max(values) if "loss" not in label else min(values)
            last_val = values[-1]
            
            findings.append(f"{label}: best={best_val:.4f}, last={last_val:.4f}")
    
    return f"{rel}: " + " | ".join(findings)
```

**实际效果**：

假设 `results/metrics.csv` 内容如下：
```csv
epoch,accuracy,loss,val_accuracy,val_loss
1,0.85,0.45,0.82,0.48
2,0.89,0.38,0.87,0.42
3,0.92,0.32,0.90,0.35
4,0.93,0.28,0.91,0.31
```

**Repo2Slides 会自动提取**：
```
results/metrics.csv: accuracy: best=0.9300 (epoch 4), last=0.9300 | loss: best=0.2800 (epoch 4), last=0.2800 | val_accuracy: best=0.9100 (epoch 4), last=0.9100
```

**设计亮点**：
- ✅ **自动识别**：无需手动指定指标列名
- ✅ **智能计算**：自动区分 loss（越小越好）和 accuracy（越大越好）
- ✅ **信息丰富**：同时提供最佳值和最后值，便于分析训练趋势

---

### 3️⃣ Slide Planner（PPT 规划器）📋

**功能**：将结构化信息转换为 8-12 页的 PPT 大纲

#### 🎯 核心能力

- **智能分页**：自动规划 Title、Outline、Background、Method、Architecture、Experiments、Results、Conclusion 等页面
- **去重优化**：自动去除重复的 bullets，合并相似页面
- **语言支持**：支持中英文两种风格的 PPT 标题和内容
- **页数控制**：确保生成的 PPT 在 8-12 页之间（符合学术汇报规范）

#### 💻 代码示例

```python
from repo2slides.planner import SlidePlanner

# 初始化规划器（可选 LLM 增强）
planner = SlidePlanner(llm_client=llm_client, language="zh")  # 或 "en"

# 规划 PPT
slide_plan = planner.plan_slides(structured_content)

print(f"共生成 {len(slide_plan)} 页 PPT")
for i, slide in enumerate(slide_plan, 1):
    print(f"\n第 {i} 页: {slide['title']}")
    for bullet in slide['bullets']:
        print(f"  - {bullet}")
```

**输出示例**：
```
共生成 10 页 PPT

第 1 页: 项目概览
  - 项目整体介绍
  - 基于仓库的代码与实验
  - 由 Repo2Slides 自动生成 PPT 大纲

第 2 页: 目录
  - 背景与动机
  - 问题定义与目标
  - 方法概述
  ...

第 3 页: 背景与动机
  - 学生 / 研究者 / 开发者，在做：
  - 课程汇报
  - 组会
  ...
```

#### 🔬 深度解析：去重算法

```python
# planner.py 中的去重逻辑
def _post_process(self, slides: List[Slide]) -> List[Slide]:
    """
    后处理：去重、合并相似页面、控制页数
    """
    cleaned: List[Slide] = []
    seen_bullets: List[str] = []
    
    for slide in slides:
        bullets = []
        for b in slide.get("bullets", []):
            # 局部去重（同一页内）
            if any(text_similarity(b, x) >= 0.92 for x in bullets):
                continue
            # 全局去重（跨页）
            if any(text_similarity(b, x) >= 0.94 for x in seen_bullets):
                continue
            bullets.append(b)
            seen_bullets.append(b)
        
        if bullets:
            cleaned.append({"title": slide["title"], "bullets": bullets})
    
    # 合并相似页面（如 Background 和 Problem 太相似）
    bg_idx = find_slide_index(cleaned, "Background")
    pr_idx = find_slide_index(cleaned, "Problem")
    if bg_idx and pr_idx:
        if text_similarity(bg_text, pr_text) >= 0.88:
            cleaned.pop(pr_idx)  # 删除重复的 Problem 页
    
    return cleaned[:12]  # 确保不超过 12 页
```

**设计亮点**：
- ✅ **智能去重**：使用文本相似度算法（token Jaccard + 序列匹配）
- ✅ **页面合并**：自动识别并合并过于相似的页面
- ✅ **页数控制**：确保输出符合学术汇报规范（8-12 页）

---

### 4️⃣ Slide Generator（PPT 生成器）🎨

**功能**：将 PPT 规划转换为 Markdown 或 PPTX 格式

#### 🎯 核心能力

- **Markdown 输出**：生成 Marp/reveal.js 兼容的 Markdown slides
- **PPTX 输出**：可选生成 PowerPoint 文件（需要 `python-pptx`）
- **Front Matter**：自动添加 YAML front matter（title、author、theme、生成时间等）
- **学术风格**：每页使用清晰的标题和 bullet points

#### 💻 代码示例

```python
from repo2slides.generator import SlideGenerator

generator = SlideGenerator()

# 生成 Markdown（带 front matter）
markdown = generator.to_markdown_with_front_matter(
    slide_plan,
    title="My Awesome Project",
    author="Repo2Slides (auto-generated)",
    theme="academic",
    engine="marp"  # 或 "plain"
)

# 保存文件
with open("slides.md", "w", encoding="utf-8") as f:
    f.write(markdown)

# 或者生成 PPTX
generator.to_pptx(slide_plan, Path("slides.pptx"))
```

**生成的 Markdown 示例**：
```markdown
---
marp: true
paginate: true
title: "My Awesome Project"
author: "Repo2Slides (auto-generated)"
theme: "academic"
generated_at: "2026-01-29T14:00:00Z"
---

# 项目概览

- 项目整体介绍
- 基于仓库的代码与实验
- 由 Repo2Slides 自动生成 PPT 大纲

---

# 目录

- 背景与动机
- 问题定义与目标
- 方法概述
- 系统架构
- 实验设置与结果
- 总结与未来工作

---
...
```

---

## 💡 使用示例

### 📚 场景 1：课程项目汇报

```bash
# 分析你的课程项目仓库
python cli.py ./my-course-project --out course_slides.md --lang zh

# 生成的 slides.md 可以直接用 Marp 打开预览
# 或者导出为 PPTX
python cli.py ./my-course-project --out course_slides.pptx --format pptx
```

### 🔬 场景 2：组会汇报（ML 项目）

```bash
# 使用 LLM 增强模式，生成更高质量的内容
export OPENAI_API_KEY="your_key"
python cli.py ./ml-research-project \
    --out group_meeting_slides.md \
    --model gpt-4o-mini \
    --lang zh \
    --verbose
```

**输出效果**：
- ✅ 自动识别 PyTorch/HuggingFace 框架
- ✅ 自动解析 `results/metrics.csv` 中的训练指标
- ✅ 自动提取 `notebooks/` 中的实验流程
- ✅ 生成符合学术规范的 PPT 大纲

### 🎓 场景 3：毕业答辩

```bash
# 生成中英文双语版本
python cli.py ./thesis-project --out thesis_slides_zh.md --lang zh
python cli.py ./thesis-project --out thesis_slides_en.md --lang en
```

---

## 🏗️ 架构设计

### 📐 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Entry Point                       │
│                  (cli.py / __main__.py)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Repo Analyzer (analyzer.py)                │
│  • 扫描仓库结构                                          │
│  • 解析 Python 文件（AST）                               │
│  • 检测 ML 框架                                          │
│  • 识别入口点和结果文件                                   │
└────────────────────┬────────────────────────────────────┘
                     │ RepoSummary
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Content Extractor (extractor.py)               │
│  • 解析 README（中英文章节识别）                         │
│  • 分析 Metrics 文件（CSV/JSON）                        │
│  • 智能推断缺失信息                                       │
│  • LLM 增强（可选）                                      │
└────────────────────┬────────────────────────────────────┘
                     │ Structured Content Dict
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Slide Planner (planner.py)                   │
│  • 规划 8-12 页 PPT                                      │
│  • 去重和页面合并                                         │
│  • 中英文风格支持                                         │
│  • LLM 优化（可选）                                      │
└────────────────────┬────────────────────────────────────┘
                     │ Slide Plan (List[Slide])
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Slide Generator (generator.py)                  │
│  • 生成 Markdown slides                                  │
│  • 生成 PPTX（可选）                                      │
│  • Front Matter 支持                                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
              slides.md / slides.pptx
```

### 🔧 核心模块详解

#### 1. **analyzer.py** - 仓库分析器

**职责**：
- 遍历仓库目录（使用 `os.walk`，可剪枝忽略目录）
- 解析 Python 文件（使用 `ast` 模块，只提取结构信息）
- 检测 ML 框架（通过 import 语句匹配）
- 识别结果文件（基于路径和文件名模式）

**关键设计**：
- ✅ **性能优化**：限制文件数量（最多 2000 个文件）和大小（512KB）
- ✅ **智能过滤**：自动忽略 `.git`、`venv`、`node_modules` 等噪声目录
- ✅ **轻量级解析**：只提取类名、函数名、docstring，不读实现代码

#### 2. **extractor.py** - 内容抽取器

**职责**：
- 解析 README（支持中英文章节识别）
- 分析 Metrics 文件（CSV/JSON 数值提取）
- 智能推断（从代码结构补全缺失信息）
- LLM 增强（可选，使用 OpenAI API）

**关键设计**：
- ✅ **双模式**：LLM 模式（高质量）和启发式模式（离线）
- ✅ **Metrics 分析**：自动识别指标列，提取最佳值和最后值
- ✅ **框架信息**：将检测到的 ML 框架自动添加到方法描述

#### 3. **planner.py** - PPT 规划器

**职责**：
- 将结构化内容转换为 PPT 页面规划
- 去重和页面合并
- 页数控制（8-12 页）
- 中英文风格支持

**关键设计**：
- ✅ **去重算法**：使用文本相似度（token Jaccard + 序列匹配）
- ✅ **页面合并**：自动识别并合并过于相似的页面
- ✅ **页数控制**：确保输出符合学术汇报规范

#### 4. **generator.py** - PPT 生成器

**职责**：
- 生成 Markdown slides（Marp/reveal.js 兼容）
- 生成 PPTX（可选，使用 `python-pptx`）
- Front Matter 支持

**关键设计**：
- ✅ **多格式支持**：Markdown 和 PPTX
- ✅ **Front Matter**：自动添加元数据（title、author、theme、生成时间）

---

## 🎛️ 命令行参数详解

### 基本参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `repo_path` | 要分析的仓库路径（位置参数） | - | `./my-repo` |
| `--out / -o` | 输出文件路径 | `slides.md` | `--out my_slides.md` |
| `--format` | 输出格式 | `md` | `--format pptx` |

### LLM 相关参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--model` | LLM 模型名 | `gpt-4o-mini` | `--model gpt-4o-mini` |
| `--api-key` | OpenAI API Key | 从环境变量读取 | `--api-key sk-...` |
| `--no-llm` | 关闭 LLM（使用纯启发式模式） | `False` | `--no-llm` |

### 样式相关参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--md-engine` | Markdown 引擎 | `marp` | `--md-engine plain` |
| `--theme` | 主题名称 | `academic` | `--theme default` |
| `--lang` | 语言风格 | `en` | `--lang zh` |

### 调试参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--verbose` | 显示详细进度信息 | `False` | `--verbose` |

### 📝 完整示例

```bash
# 使用 LLM 增强 + 中文风格 + 详细日志
python cli.py ./my-repo \
    --out slides.md \
    --model gpt-4o-mini \
    --api-key sk-your-key \
    --lang zh \
    --md-engine marp \
    --theme academic \
    --verbose

# 纯离线模式（无需 API Key）
python cli.py ./my-repo \
    --out slides.md \
    --no-llm \
    --lang zh
```

---

## 🔍 技术细节

### 🧠 LLM 使用策略

**Repo2Slides 的 LLM 使用原则**：

1. **不直接读代码**：只使用文件路径、类名、函数名、docstring
2. **结构化输出**：强制 LLM 输出 JSON 格式的结构化内容
3. **容错处理**：LLM 失败时自动回退到启发式模式
4. **Token 限制**：自动截断过长的 README 和代码摘要

**Prompt 设计**：

```python
# prompts.py 中的系统提示词
SYSTEM_SUMMARIZE_README = """
You are an expert research assistant helping to prepare an academic-style
presentation for a software / ML project.

Your job:
1. Understand the project background & motivation.
2. Identify the main problem and goals.
3. Summarize the method / algorithm / system design.
4. Infer the architecture (modules, data flow) at a high level.
5. Extract experiment setup and metrics (if any).
6. Summarize experiment results and key findings.
7. Propose a short conclusion and possible future work.

Output must be a JSON object with the following keys:
- "title", "background", "problem", "method", "architecture",
  "experiments", "results", "conclusion", "future_work"
"""
```

### 🚀 性能优化

1. **文件数量限制**：最多分析 2000 个文件，避免大仓库卡死
2. **文件大小限制**：单个文件最大 512KB，跳过过大文件
3. **Python 文件限制**：最多解析 300 个 Python 文件
4. **目录剪枝**：自动跳过 `.git`、`venv`、`node_modules` 等目录

### 🛡️ 错误处理

- **LLM 失败**：自动回退到启发式模式
- **文件读取失败**：跳过无法读取的文件，继续处理其他文件
- **JSON 解析失败**：尝试从响应中提取 JSON 块
- **Metrics 解析失败**：回退到简单的列名/键名显示

---

## 📚 开发者指南

### 🏗️ 项目结构

```
repo2slides/
├── README.md                 # 项目文档（本文件）
├── requirements.txt          # Python 依赖
├── cli.py                    # 命令行入口（兼容旧版本）
├── repo2slides/
│   ├── __init__.py          # 包初始化
│   ├── __main__.py          # Python -m 入口
│   ├── cli_entry.py         # CLI 逻辑（统一入口）
│   ├── analyzer.py          # 仓库分析器
│   ├── extractor.py         # 内容抽取器
│   ├── planner.py           # PPT 规划器
│   ├── generator.py         # PPT 生成器
│   ├── utils.py             # 工具函数（LLM 封装、文本处理等）
│   └── prompts.py           # LLM prompt 模板
└── examples/
    ├── sample_output.md     # 示例输出
    └── self_slides_*.md     # 自测生成的 slides
```

### 🔨 开发流程

1. **本地开发**：
   ```bash
   # 克隆仓库
   git clone https://github.com/yourname/Repo2Slides.git
   cd Repo2Slides
   
   # 安装开发依赖
   pip install -r requirements.txt
   
   # 运行自测
   python cli.py . --out examples/self_test.md --no-llm --verbose
   ```

2. **添加新功能**：
   - 修改对应模块（`analyzer.py`、`extractor.py` 等）
   - 运行自测确保功能正常
   - 更新 README 文档

3. **测试 LLM 功能**：
   ```bash
   export OPENAI_API_KEY="your_key"
   python cli.py ./test-repo --model gpt-4o-mini --verbose
   ```

### 🧪 测试建议

1. **小型仓库测试**：先用一个只有 README + 1-2 个 Python 文件的小仓库测试
2. **检查分析结果**：确认 `RepoAnalyzer.analyze()` 的输出合理
3. **检查抽取结果**：确认 `ContentExtractor.extract()` 的结构化 dict 完整
4. **检查 PPT 规划**：确认 `SlidePlanner.plan_slides()` 的页数和内容合理
5. **预览输出**：用 Marp 或 Reveal.js 打开生成的 markdown 文件预览

---

## 🎯 未来规划

### 🚀 即将支持的功能

- [ ] **更多语言栈支持**：Rust、Go、Java、TypeScript 等
- [ ] **Diagram 自动生成**：自动画模型结构图、实验流程图
- [ ] **PPTX 模板支持**：支持自定义 PowerPoint 模板
- [ ] **批量处理**：支持一次处理多个仓库
- [ ] **Web UI**：提供 Web 界面，无需命令行

### 💡 改进方向

- [ ] **更智能的代码理解**：使用代码嵌入向量进行语义相似度匹配
- [ ] **实验结果可视化**：自动生成图表（accuracy curve、loss curve 等）
- [ ] **多语言 PPT**：支持生成中英文混合的 PPT
- [ ] **自定义 Prompt**：允许用户自定义 LLM prompt 模板

---

## 🤝 贡献指南

欢迎贡献代码！🎉

1. **Fork 本仓库**
2. **创建特性分支**：`git checkout -b feature/amazing-feature`
3. **提交更改**：`git commit -m 'Add some amazing feature'`
4. **推送到分支**：`git push origin feature/amazing-feature`
5. **提交 Pull Request**

### 📝 代码规范

- 使用 Python 3.9+ 类型提示
- 遵循 PEP 8 代码风格
- 添加必要的 docstring
- 确保新功能有对应的测试

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 许可证。

---

## 🙏 致谢

- [Marp](https://marp.app/) - Markdown 幻灯片工具
- [python-pptx](https://python-pptx.readthedocs.io/) - PowerPoint 文件生成
- [OpenAI](https://openai.com/) - LLM API 支持

---

## 📞 联系方式

- **Issues**：https://github.com/yourname/Repo2Slides/issues
- **Discussions**：https://github.com/yourname/Repo2Slides/discussions

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！⭐**

Made with ❤️ by the Repo2Slides team

</div>
