# Shiyi 记忆系统设计（2026-02-12）

## 1. 目标与范围

本期目标是把 Shiyi 从“会话可持久化”升级为“可检索、可进化、可代谢”的长期记忆系统。  
核心方向：

- 采用三层记忆漏斗：`L0 原始层(SQLite)`、`L1 认知层(Markdown)`、`L2 检索层(sql-vec + FTS5)`。
- 用户建模为**全局唯一用户**，所有会话共享同一份 `User.md`。
- 首次对话触发身份引导（定义十一身份 + 确认用户身份），确认后不再重复询问。
- 每轮固定注入精简用户卡片；长期记忆采用“常驻近期 + 按需长期检索”。
- 写入采用置信度分流，高置信自动写入，低置信以内联交互确认。

非目标（本期不做）：

- 多用户隔离与租户级权限系统（只保留可扩展结构）。
- 分布式向量数据库集群（先以 SQLite 生态能力为主）。

---

## 2. 总体架构（三层漏斗）

### 2.1 L0 原始层（SQLite：全量事实）

职责：完整记录所有原始交互与工具行为，作为审计与可追溯事实源。  
主要数据：

- `sessions` / `messages`（现有会话与消息）
- `tool_logs`（工具调用与结果）
- `token_usage`（成本与上下文消耗）
- `memory_events`（记忆更新流水、检索失败、重试日志）

特点：

- 只增不改（业务层），偏事实存档。
- 所有高层记忆都可追溯到原始 `source_message_id`。

### 2.2 L1 认知层（Markdown：系统可读、人工可维护）

职责：维护会被高频注入模型的精炼认知。  
文件布局：

- `data/memory/system/ShiYi.md`：十一人设与行为边界
- `data/memory/shared/User.md`：全局唯一用户画像
- `data/memory/shared/Project.md`：项目进度与滚动总结
- `data/memory/shared/Insights.md`：热点经验池（仅热数据）

特点：

- 不是流水账，必须执行“记忆代谢”。
- 支持人工审阅与快速修订。

### 2.3 L2 检索层（sql-vec + FTS5：长期召回）

职责：从海量历史中按语义和关键词召回高相关片段。  
策略：

- 语义检索：`sql-vec`（embedding 相似度）
- 精确检索：`FTS5 + SQL`（关键词、文件名、日期、统计）
- 混合重排：语义分 + 关键词分 + 新鲜度分

---

## 3. 首次身份引导（仅一次）

系统维护 `users.identity_confirmed`（全局用户记录，`user_id='global'`）。

当 `identity_confirmed=false` 时触发 onboarding：

1. 引导定义十一身份：身份定位、语气、边界、禁区、优先行为。
2. 确认用户身份：称呼、职业背景、技术偏好、协作习惯。
3. 输出确认摘要，用户可“确认/修改”。
4. 确认后写入：
   - `ShiYi.md`
   - `User.md`
   - `users.identity_confirmed=true`

后续会话不再重复询问，仅通过反思与低干扰确认机制增量更新。

---

## 4. 数据模型设计

在现有表基础上新增（可用 Alembic 迁移）：

- `users`
  - `user_id`（PK，固定 `global`）
  - `identity_confirmed`（bool）
  - `display_name`
  - `created_at`, `updated_at`
- `memory_facts`
  - `id`（PK）
  - `scope`（system/user/project/insight）
  - `fact_type`（identity/preference/constraint/experience/...）
  - `fact_key`, `fact_value`
  - `confidence`（0-1）
  - `status`（active/superseded/rejected）
  - `source_message_id`
  - `created_at`, `updated_at`
- `memory_pending`
  - `id`（PK）
  - `candidate_fact`（JSON）
  - `confidence`
  - `status`（pending/confirmed/rejected/snoozed）
  - `cooldown_until`
  - `source_message_id`
  - `created_at`, `updated_at`
- `memory_embeddings`
  - `id`（PK）
  - `source_type`（fact/message/insight）
  - `source_id`
  - `embedding`（sql-vec 列）
  - `created_at`
- `memory_events`
  - `id`（PK）
  - `event_type`（write/update/conflict/retrieval_fail/retry/...）
  - `operation_id`（幂等键）
  - `payload`（JSON）
  - `created_at`

兼容策略：

- `sessions.user_id` 统一关联 `global`。
- 保留未来多用户扩展路径，不破坏当前单用户体验。

---

## 5. 读取编排（常驻近期 + 按需长期）

在 `AgentCore` 前增加 Memory Orchestrator，执行固定顺序：

1. 组装系统记忆注入包  
   - 从 `ShiYi.md + User.md` 编译精简卡片（建议 300-500 tokens）。
2. 注入近期上下文  
   - 最近 8-12 轮会话（或 token 限额内最大窗口）。
3. 判断是否触发长期检索（按需）  
   - 触发条件：用户提及“之前/还记得/上次/历史/某日期/某文件”等。
4. 混合检索  
   - `sql-vec` 与 `FTS5` 并行召回（各 topK=20）。
   - 重排公式建议：`0.55*semantic + 0.30*keyword + 0.15*freshness`。
   - 去重后取 3-5 条证据注入。
5. 精确查询分流  
   - 用户明确日期、字段、统计需求时，走 SQL 精确查询工具。

Token 预算（`max_context_tokens=4000`）建议：

- 系统记忆：600
- 近期上下文：2000
- 长期证据：900
- 生成预留：500

超限裁剪优先级：

1. 先裁剪长期证据
2. 再压缩近期对话
3. 最后保留 `ShiYi/User` 核心卡片不动

---

## 6. 写入与代谢编排

每轮会话后触发 `summarize_and_store`（增量处理）：

1. 抽取候选事实（persona/user/project/insight 分类）。
2. 置信度分流：
   - `>=0.85`：自动写入
   - `0.60~0.85`：写入 `memory_pending`
   - `<0.60`：丢弃
3. 冲突处理（覆盖策略）  
   - 基于“新近性 + 一致性命中次数”做覆盖或保留。
4. Markdown 原子更新  
   - 临时文件写入 -> rename 替换正式文件。
5. 向量异步入库  
   - 对确认后的事实/片段生成 embedding 写入 `memory_embeddings`。

代谢规则：

- `User.md`：覆盖式更新，保持名片化（建议 <=2KB）。
- `Project.md`：超过阈值（如 >100 行）触发滚动总结归档。
- `Insights.md`：只保热点（如最近常用 10 条），其余下沉 DB。

---

## 7. 内联确认交互（低置信记忆）

当本轮产生 `pending` 记忆时，助手在回复末尾附最多 1-3 条候选：

- 动作：`确认` / `忽略` / `稍后`

状态流转：

- `确认`：`pending -> confirmed`，更新 Markdown + 向量入库
- `忽略`：`pending -> rejected`，避免重复打扰
- `稍后`：`pending -> snoozed`，设置 `cooldown_until`

交互目标：

- 把“写错记忆”成本压低
- 把“确认正确记忆”动作前置到对话当下

---

## 8. 稳定性与异常治理

幂等与一致性：

- 每次写入生成 `operation_id`，重复执行不重复落库。
- DB 写操作放事务中，确保事实与状态一致。

降级链路：

1. `sql-vec` 失败 -> 降级 FTS5
2. FTS5 失败 -> 仅使用近期对话与 Markdown 核心卡片
3. 记录 `memory_events(event_type='retrieval_fail')`

异步任务可靠性：

- 向量入库队列重试（指数退避）
- 超过重试阈值进入 dead-letter，并保留人工修复入口

---

## 9. 验收与测试

测试分层：

- 单元测试：置信度分流、冲突覆盖、代谢规则、状态流转。
- 集成测试：DB/Markdown/向量全链路写读一致性。
- 端到端：首次身份引导只触发一次；后续会话可复用记忆。
- 故障测试：模拟 sql-vec/FTS5 故障，验证降级可用。

核心指标（建议）：

- 记忆确认率（confirmed / pending）
- 错误写入率（rejected / confirmed）
- 长期检索命中率（有用证据占比）
- 上下文记忆 token 占比（长期稳定）

---

## 10. 里程碑与 MVP

### M0 数据底座

- 新表与迁移脚本
- 兼容现有 sessions/messages

### M1 首次身份引导

- onboarding 状态机
- 写入 `ShiYi.md` 与共享 `User.md`

### M2 读取编排

- 常驻近期 + 按需长期检索
- 先接 FTS5，再接 sql-vec 混合重排

### M3 写入与代谢

- `summarize_and_store`
- 置信度分流 + 内联确认 + 原子写文件

### M4 稳定性与观测

- 幂等、重试、降级、指标面板

MVP 建议：`M0 + M1 + M2(FTS5先行) + M3(基础版)`，优先拿到“可感知的长期记忆能力”。
