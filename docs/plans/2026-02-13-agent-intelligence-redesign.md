# Agent 智能重设计方案

**日期**：2026-02-13
**状态**：已确认，待实施
**背景**：现有 GPT 重构版本（Router-Planner-Policy 五层架构）过度工程化，引入额外延迟且未实质提升智能。本方案推翻重来，以"LLM 自主决策 + 优质上下文供给"为核心哲学重新设计。

---

## 核心哲学

| 旧设计 | 新设计 |
|--------|--------|
| 规则系统替 LLM 做决定 | LLM 自主决策，系统提供优质原料 |
| 静态记忆注入（全量 Markdown 文件） | 动态记忆工具（LLM 主动查询 `query_memory`） |
| 每轮多一次 LLM 分类 intent（额外 API 调用） | 启发式规则检测复杂度（零 API 调用） |
| SQLite 承担所有记忆职责 | Kuzu 负责智能检索，SQLite 只存原始日志 |

---

## 顶层数据流

```
用户输入
  ↓
① Context Builder（纯本地，零 LLM 调用）
    ├─ 人设 system prompt（ShiYi.md + 当前时间 + 工具使用指引）
    ├─ 滑动窗口会话历史（token 预算：2000 tokens，超 20 轮生成摘要）
    └─ Kuzu 关键词预查询：注入 1-2 条高置信 Fact 节点（线索，非全量）
  ↓
② Complexity Detector（启发式规则，零 LLM 调用）
    ├─ 简单 → 直接进 ReAct 循环
    └─ 复杂 → 在 system prompt 追加计划指引 → ReAct 循环
  ↓
③ ReAct 循环（LLM 完全自主决策）
    ├─ query_memory(query, mode)  → Kuzu 图谱遍历 + 向量精排
    ├─ search_web(query)          → 实时信息
    ├─ bash(cmd)                  → 工作区操作
    ├─ read_file / write_file / edit_file → 文件操作
    └─ 无工具 → 直接回复
  ↓
④ 异步记忆写入（回复后 fire-and-forget）
    ├─ Event 节点 → Kuzu 事件流（立即写，无 LLM）
    ├─ embedding 计算 → 更新 Event.embedding（队列）
    ├─ LLM 实体抽取 → 更新 Kuzu 图谱关系（队列）
    └─ 原始消息 → SQLite（session 恢复用）
```

---

## 记忆架构：Kuzu + SQLite 双层

### 数据分层职责

```
┌─────────────────────────────────────────────────────┐
│  Kuzu Graph DB（智能层，主要查询入口）                 │
│                                                      │
│  事件流节点                  图谱关系节点              │
│  Event(                     Entity(name, type)       │
│    id, session_id,          Relation(                │
│    timestamp, role,           from, to, type,        │
│    content, summary,          weight)                │
│    embedding[])             Fact(                    │
│                               key, value,            │
│                               scope, confidence,     │
│                               updated_at)            │
│                                                      │
│  查询方式：                                           │
│  - 向量相似度（embedding 字段，高维）                  │
│  - 图遍历（实体关系跳跃，Cypher 语法）                 │
│  - 混合查询（semantic 初筛 → 图遍历扩展）              │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  SQLite（原始日志层，只写不查）                        │
│  sessions / messages（会话恢复用）                    │
│  embedding_jobs（异步任务队列）                       │
└─────────────────────────────────────────────────────┘
```

### Kuzu Schema（Cypher）

```cypher
-- 节点
CREATE NODE TABLE Event(
  id STRING PRIMARY KEY,
  session_id STRING,
  timestamp INT64,
  role STRING,
  content STRING,
  summary STRING,
  embedding FLOAT[1536]
)

CREATE NODE TABLE Entity(
  name STRING PRIMARY KEY,
  type STRING    -- person / technology / project / concept
)

CREATE NODE TABLE Fact(
  id STRING PRIMARY KEY,
  key STRING,
  value STRING,
  scope STRING,  -- user / system / project
  confidence FLOAT,
  updated_at INT64
)

CREATE NODE TABLE Session(
  id STRING PRIMARY KEY,
  title STRING,
  created_at INT64
)

-- 关系
CREATE REL TABLE IN_SESSION(FROM Event TO Session)
CREATE REL TABLE NEXT(FROM Event TO Event)
CREATE REL TABLE MENTIONS(FROM Event TO Entity)
CREATE REL TABLE RELATED_TO(FROM Entity TO Entity, type STRING, weight FLOAT)
CREATE REL TABLE ABOUT(FROM Fact TO Entity)
```

### `query_memory` 工具三种模式

| mode | 触发场景 | 执行逻辑 |
|------|---------|---------|
| `semantic` | "上次我们聊过什么" / 模糊回忆 | embedding 相似度检索 Event 节点，返回 top-5 事件摘要 |
| `graph` | "我和这个项目有什么关系" / 实体推理 | 从 query 提取实体名 → Kuzu 图遍历 2 跳 → 返回关联 Fact + Relation |
| `hybrid` | 默认模式 | semantic 初筛 → 取涉及 Entity → 图遍历扩展 → 融合排序 |

### 异步写入流水线

```
回复完成 → fire-and-forget
  ①  写 Event 节点（立即，无 LLM）
  ②  embedding 计算 → 更新 Event.embedding（队列）
  ③  LLM 实体抽取（temperature=0，输入最近 3 条对话）
      输出：[{entity, type, relations[], facts[]}]
      confidence >= 0.75 → 直接写图谱节点
      0.55 ~ 0.74       → pending，待合并
```

---

## Context Builder

```python
# token 预算分配（总计约 6000 tokens）
TokenBudget:
  system_prompt:    ~500 tokens   # 人设 + 时间 + 工具使用指引（固定）
  kuzu_prefetch:    ~200 tokens   # 关键词匹配 1-2 条高置信 Fact（固定）
  session_summary:  15% of 余量   # 超 20 轮对话后自动生成摘要替代头部历史
  history_window:   85% of 余量   # 滑动窗口，tail_first 策略（保留最近）
```

**`kuzu_prefetch` 设计原则**：不靠 LLM 猜测注入什么记忆，只做关键词匹配拿 1-2 条线索，让 LLM 看到"有记忆可查"后自主调用 `query_memory` 深入检索。

---

## Complexity Detector

### 触发条件（满足任意一条 → 复杂任务模式）

全部条件通过 `config/config.yaml` 配置，不硬编码：

```yaml
complexity_detector:
  enabled: true
  step_keywords:
    - "步骤"
    - "先.*再.*然后"
    - "分析并"
    - "重构"
    - "迁移"
    - "帮我做"
    - "帮我搞"
  multi_tool_domains:
    search: ["搜索", "查一下", "找找"]
    file:   ["文件", "代码", "读取"]
    shell:  ["执行", "运行", "命令"]
  multi_tool_threshold: 2       # 同时出现几个领域算复杂
  message_length_threshold: 80  # 消息字数超过此值
  continuation_markers:         # 上轮助手回复包含这些标记
    - "接下来"
    - "第一步"
    - "下一步"
```

### 触发后行为

**不调用 LLM**，在 system prompt 追加：

```
[任务规划模式]
这是一个多步任务。请先在回复开头列出执行计划：
  计划：
  1. <步骤一>
  2. <步骤二>
  ...
然后立即开始执行，无需等待用户确认。
```

LLM 自主生成计划并执行，无需额外 API 调用。

---

## TUI 进度展示

工具调用过程显示简短进度行，完成后保留结果摘要：

```
用户：帮我分析 shiyi 项目的性能瓶颈并给出优化方案

妲己：
  计划：
  1. 读取核心模块代码
  2. 分析性能热点
  3. 给出优化建议

  [1/3 读取文件...]  ████░░░░░░
  [2/3 分析中...]    ████████░░
  [3/3 生成建议...]  完成 ✓

  <最终回复正文>
```

---

## 分阶段实施路线

### Phase 1：清理 + 解放 LLM

**目标**：删掉行政审批层，LLM 自由调用工具，context 有预算控制。

```
删除：
  - core/agent_orchestration.py 中的 OrchestrationRouter
  - PolicyScheduler（工具白名单限制）
  - EvidenceCollector（Evidence block）
  - agent_core.py 中所有 orchestration 钩子

保留并改造：
  - core/agent_core.py → 精简为纯 ReAct 循环
  - core/session_manager.py → 加 token 预算 + 滑动窗口截断

新增：
  - core/context_builder.py       → TokenBudget + sliding_window
  - core/complexity_detector.py  → 读 config 的启发式规则
  - config.yaml → complexity_detector 配置块
```

**验收标准**：
- 工具调用无白名单限制，LLM 可调用所有工具
- 会话历史超过 token 预算时自动截断，不报错
- 复杂任务触发计划模式

---

### Phase 2：Kuzu 记忆层

**目标**：接入 Kuzu，SQLite 退化为日志，`query_memory` 工具上线。

```
新增：
  - memory/kuzu_store.py       → Kuzu 初始化 + schema 建表
  - memory/kuzu_writer.py      → 异步写 Event/Entity/Fact 节点
  - memory/kuzu_retriever.py   → semantic/graph/hybrid 三种查询
  - tools/builtin/query_memory.py → LLM 可调用的记忆工具

改造：
  - session_manager.py → 写操作路由到 Kuzu
  - storage.py         → 只保留 sessions/messages 表
  - documents.py       → memory_card 改为从 Kuzu 查高置信 Fact

数据迁移：
  - memory_facts / memory_embeddings → 导入 Kuzu
```

**验收标准**：
- `query_memory` 三种模式均可调用并返回有意义结果
- 新对话内容异步写入 Kuzu，重启后可检索

---

### Phase 3：图谱推理 + TUI 进度

**目标**：实体关系图谱跑通，TUI 展示计划进度。

```
新增：
  - memory/entity_extractor.py → 异步 LLM 实体抽取
  - memory/kuzu_graph.py       → 图遍历 2 跳 Cypher 查询
  - channels/cli/progress.py  → TUI 进度条渲染

打通：
  - query_memory mode="graph" 完整链路
  - Complexity Detector → 计划生成 → TUI 进度步骤标注
```

**验收标准**：
- 实体关系图谱随对话自动增长
- 图遍历查询可跨实体返回关联信息
- 复杂任务在 TUI 显示步骤进度

---

## 架构演进对比

```
当前：  用户 → Router(LLM) → PolicyScheduler → Planner → ReAct → SQLite(全部)

P1后：  用户 → ContextBuilder → ReAct（工具自由）→ SQLite

P2后：  用户 → ContextBuilder → ReAct（+query_memory）→ Kuzu + SQLite

P3后：  用户 → ContextBuilder → ComplexityDetector → ReAct（图谱记忆）→ Kuzu + SQLite
```

---

## 非目标（YAGNI）

- 不引入向量数据库（Kuzu 内置向量即可）
- 不重写多 agent 委派系统
- 不改变 channels 层（CLI/API/Voice 保持现有接口）
- 不替换 LLM 引擎（DeepSeek OpenAI-compatible 继续使用）
