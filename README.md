# 基于LLM Agent决策的自适应用户画像与行为预测系统（UserInsight Agent）

自主规划式用户智能分析 Agent —— 基于 **LangGraph** 的 LLM Agent，面向数据分析 / 数据科学岗位的项目。

用户用一句自然语言描述分析目标，Agent 自主完成：**任务规划 → 工具调用 → 反思重试 → 业务报告** 的完整闭环，输出用户分群画像、流失预测、图表与双格式报告。

## 核心能力

### 1. 自主规划（Planner）
LLM 将自然语言目标分解为带依赖关系的工具调用计划（DAG），并从 ChromaDB 长期记忆中召回历史 schema 与有效策略作为规划上下文。

### 2. 工具调用（Executor）
9 类分析工具通过 Function Calling 暴露给 LLM，由 Agent 自主选择工具与参数：
`load_data` / `inspect_schema` / `clean_data` / `engineer_features` / `segment_users` / `train_predictor` / `plot_charts` / `generate_report` / `quality_check`

### 3. 反思循环（Critic）
每步执行后评估结果与数据质量指标，未达标自动回溯重试（清洗-诊断闭环、分群-评估闭环、预测-基线闭环）。以最大迭代轮数与重试预算为护栏，保证可控终止。

### 4. 长期记忆（ChromaDB）
持久化历史 schema、有效策略与业务结论，跨会话复用经验；chromadb 不可用时自动降级为 JSON 文件存储。

### 5. 脏数据自适应处理
- 编码自适应探测（utf-8-sig / utf-8 / gbk / gb2312 / latin-1）
- 列名清洗（去空、去重、去注入风险字符）
- 缺失值智能填充（数值→中位数，类别→众数）
- **高缺失率列（>80%）自动剔除**，避免单值填充引入假信号
- 常量列（零方差）自动剔除、inf 值自动转 NaN
- 4 类字段类型自动识别（时间/分类/数值/ID）
- 聚类四算法（K-Means / GMM / 层次 / DBSCAN）按轮廓系数自动择优，DBSCAN 带噪声率与规模护栏
- 分类四模型（LR / RF / GBDT / SVM）按分层交叉验证自动选模，不平衡标签自动加平衡权重

### 6. 优雅降级
无 API Key 时每个 LLM 节点回退为确定性策略（静态计划 + 规则 Critic），全流程仍可运行。

## 架构

```
交互层   Streamlit（自然语言目标 + 7 Tab 结果浏览）
编排层   LangGraph 状态图: Planner → Executor ⇄ Critic → Summarizer
工具层   Function Calling 注册的 9 个分析工具（共享 Workspace）
记忆层   ChromaDB 向量库（schema / strategy / insight）+ JSON 兜底
数据层   Pandas / scikit-learn / matplotlib
```

## 项目结构

```
userinsight-agent/
├── main.py                    # 入口：LangGraph agent 运行
├── app.py                     # Streamlit Web 界面
├── config.yaml                # 配置（LLM / agent 护栏 / 清洗阈值）
├── requirements.txt
├── src/
│   ├── agent_graph.py         # LangGraph 图（planner/executor/critic/summarize）
│   ├── agent_tools.py         # 9 个 Function Calling 工具
│   ├── graph_state.py         # AgentState 状态定义
│   ├── memory_store.py        # ChromaDB 长期记忆（JSON 兜底）
│   ├── data_loader.py         # 加载 + 列名清洗 + 边界处理
│   ├── data_cleaner.py        # 清洗 + 高缺失列策略
│   ├── feature_engineering.py # 特征工程（自动类型识别）
│   ├── user_segmentation.py   # 四算法自动择优分群（K-Means/GMM/层次/DBSCAN）
│   ├── behavior_prediction.py # 流失预测（LR/RF/GBDT/SVM 自动选模）
│   ├── visualization.py       # 7 类图表
│   └── report_generator.py    # Excel / HTML / 控制台报告
├── data/                      # 输入数据（含示例）
├── output/                    # 报告、图表、记忆
└── docs/
    └── interview-qa.md        # 面试追问备答
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（可选，无 Key 自动降级）
set DEEPSEEK_API_KEY=sk-...

# 3. 命令行运行（使用示例数据）
python main.py

# 4. Web 界面
streamlit run app.py
```

## 技术栈

- **LangGraph / langchain-openai** — Agent 编排与 Function Calling
- **DeepSeek LLM**（OpenAI 兼容，可切任意兼容端点 / 本地模型）
- **ChromaDB** — 长期向量记忆
- **pandas / numpy / scikit-learn** — 数据处理与建模
- **matplotlib / seaborn** — 可视化
- **Streamlit** — Web 交互界面
