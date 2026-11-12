# UserInsight Agent — 架构与流程指南

## 1. Agent 能做什么

输入任意表格型用户行为数据（CSV/Excel）+ 一句自然语言目标，Agent 自主完成：

1. **规划（Planner）**：分解目标为工具调用计划，召回 ChromaDB 长期记忆
2. **执行（Executor）**：通过 Function Calling 自主选择 9 类工具及参数
3. **反思（Critic）**：评估每步结果与数据质量，不达标自动回溯重试
4. **汇总（Summarizer）**：生成业务报告 + 写入长期记忆

**无需人工配置** — 列类型、编码、模型选择全部自适应。

---

## 2. Agent 图结构（LangGraph）

```
                ┌────────────────────────────┐
                │        Planner (LLM)        │
                │  目标 → 任务计划 DAG + 记忆    │
                └────────────┬───────────────┘
                             ▼
                ┌────────────────────────────┐
        ┌──────→│       Executor (LLM)        │
        │       │  Function Calling 选工具调参  │
        │       └────────────┬───────────────┘
        │                    ▼
        │       ┌────────────────────────────┐
        │       │        Critic (LLM)         │
        │       │  评估结果 vs 成功标准         │
        │       └─────┬──────────────┬───────┘
        │     retry  │              │ pass
        └────────────┘              ▼
                     ┌────────────────────────────┐
                     │      Summarizer (LLM)       │
                     │  业务报告 + 写入 ChromaDB     │
                     └────────────────────────────┘
```

- **循环护栏**：`agent.max_iterations`（总轮数上限）+ 每步重试预算 3 次
- **确定性兜底**：无 API Key 时 Planner 回退静态计划、Critic 回退规则判断，图照常运行

---

## 3. 工具清单（Function Calling）

| 工具 | 功能 | 关键参数 |
|---|---|---|
| `load_data` | 加载 CSV/Excel | file_path |
| `inspect_schema` | 结构摘要（dtype/唯一值/缺失率） | — |
| `clean_data` | 去重/填充/离群剔除 | fill_strategy, drop_outliers |
| `engineer_features` | 类型识别/编码/缩放/时间特征 | id_column |
| `segment_users` | 四算法自动择优分群（K-Means/GMM/层次/DBSCAN，轮廓系数评选） | n_clusters（0=肘部自动）, method |
| `train_predictor` | 四模型交叉验证选模（LR/RF/GBDT/SVM） | target_column, model |
| `plot_charts` | 7 类图表 | — |
| `generate_report` | Excel + HTML 报告 | — |
| `quality_check` | 数据质量指标（供 Critic） | — |

工具在共享 Workspace 上操作；LLM 只传标量参数，从不传数据。

---

## 4. 脏数据边界处理

### 加载层（data_loader.py）
| 边界情况 | 处理 |
|---|---|
| 未知编码 | 6 级编码链探测（utf-8-sig → utf-8 → gbk → gb2312 → latin-1 → iso-8859-1） |
| 空文件 / 全空数据 | 明确 ValueError，不崩溃 |
| 重复/空白/注入字符列名 | 清洗为安全名（去空、去重、去特殊字符） |
| 全空行/全空列 | 自动剔除并告警 |

### 清洗层（data_cleaner.py）
| 边界情况 | 处理 |
|---|---|
| 高缺失率列（>80%） | 自动剔除（单值填充会注入假信号） |
| 常量列（零方差） | 自动剔除（无信息） |
| inf / -inf | 自动转 NaN 再填充 |
| 数值列缺失 | 中位数填充；类别列缺失 → 众数填充 |
| 全部列被剔除 | 明确报错"数据不可用" |

---

## 5. 记忆体系（ChromaDB）

| 类型 | 内容 | 用途 |
|---|---|---|
| `schema_*` | 数据集结构 + 质量要点 | 相似数据复用清洗经验 |
| `strategy_*` | 有效的清洗/预测策略记录 | Planner 规划上下文 |
| `insight_*` | 历史业务结论 | Summarizer 参考 |

chromadb 不可用时自动降级为 `output/memory/memory.json`（关键词匹配召回）。

---

## 6. 运行方式

```bash
# CLI（无 Key 自动降级为确定性流水线）
python main.py

# 指定数据与目标（程序化调用）
from main import main_pipeline
result = main_pipeline(
    file_path="data/sample_user_data.csv",
    goal="找出高价值流失风险用户",
    api_key=None,          # 或传入 DeepSeek Key
    max_iterations=15,     # 循环护栏
)

# Web 界面
streamlit run app.py
```

---

## 7. 输出物

- 7 类图表：分群分布 / 分群画像 / 相关性热力图 / 特征重要性 / 混淆矩阵 / 预测概率分布 / 模型性能卡
- Excel 报告（分群/画像/重要性/指标 4 工作表）+ 自包含 HTML 报告
- Agent 执行轨迹：任务计划、每步工具结果、Critic 决策（UI「执行轨迹」Tab）
- 长期记忆落盘：`output/memory/`
