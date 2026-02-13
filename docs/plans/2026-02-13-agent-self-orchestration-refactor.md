# Agent Self-Orchestration Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the current single-loop ReAct agent into a router-planner-policy-executor-verifier architecture that self-schedules tool usage and always returns traceable evidence.

**Architecture:** Introduce a deterministic orchestration layer before and after the LLM tool loop. The router classifies user intent, policy constrains tool budget and allowlist, planner builds a lightweight executable plan, executor runs with filtered tools, and verifier enforces evidence-backed final output. Existing memory extraction and RAG injection remain in `SessionManager` and are reused as first-class evidence context.

**Tech Stack:** Python 3.11, asyncio, existing `AgentCore`, `OpenAICompatibleEngine`, `ToolRegistry`, pytest/pytest-asyncio.

---

### Task 1: Add Orchestration Domain Layer

**Files:**
- Create: `core/agent_orchestration.py`
- Test: `tests/test_agent_orchestration.py`

**Step 1: Write the failing test**

Add tests for:
- intent routing (`chat`, `memory`, `realtime_info`, `workspace_action`)
- policy tool allowlist selection
- planner output shape (`plan_id`, `steps`, `requires_tools`)
- evidence collector summary rendering

**Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest -q tests/test_agent_orchestration.py`
Expected: FAIL with missing module/classes.

**Step 3: Write minimal implementation**

Implement in `core/agent_orchestration.py`:
- `IntentType` enum
- `IntentRoute` dataclass
- `ExecutionPolicy` dataclass
- `ExecutionPlan` dataclass
- `EvidenceItem` dataclass
- `OrchestrationRouter.route(messages)`
- `PolicyScheduler.build(route)`
- `LightweightPlanner.build(route, policy, messages)`
- `EvidenceCollector.add_tool_evidence(...)`, `.render_summary()`

**Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest -q tests/test_agent_orchestration.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add core/agent_orchestration.py tests/test_agent_orchestration.py
git commit -m "feat: add agent orchestration domain layer"
```

### Task 2: Refactor AgentCore to Router + Planner + Policy + Verifier

**Files:**
- Modify: `core/agent_core.py`
- Test: `tests/test_agent_core_orchestration.py`

**Step 1: Write the failing test**

Add tests for:
- disables tools for chat/memory intents
- narrows tools to search-only for realtime info
- narrows tools to workspace tools for coding/workspace intent
- adds evidence section when tools were used

Use monkeypatch/fake engine stream so tests do not hit external APIs.

**Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest -q tests/test_agent_core_orchestration.py`
Expected: FAIL because current `AgentCore` has no orchestration hooks.

**Step 3: Write minimal implementation**

In `core/agent_core.py`:
- initialize router/scheduler/planner/evidence collector
- pre-execution route + policy + plan construction
- filter tool definitions by policy allowlist
- inject plan summary into system context for executor
- collect tool evidence on each `tool_result`
- post-execution verifier:
  - if tools used and no evidence in final answer, append standardized evidence block
  - preserve streaming behavior and existing event contract

**Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest -q tests/test_agent_core_orchestration.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add core/agent_core.py tests/test_agent_core_orchestration.py
git commit -m "feat: refactor agent core with router-planner-policy-verifier"
```

### Task 3: Expose Configurable Orchestration Knobs

**Files:**
- Modify: `config/settings.py`
- Modify: `config/config.yaml`
- Test: `tests/test_settings_orchestration.py`

**Step 1: Write the failing test**

Add tests for:
- new `agent.orchestration` config fields parse correctly
- defaults are backward compatible

**Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest -q tests/test_settings_orchestration.py`
Expected: FAIL because fields do not exist.

**Step 3: Write minimal implementation**

Add nested config model for orchestration:
- `enabled`
- `max_plan_steps`
- `force_evidence_section`
- `tool_budget_by_intent`

Wire defaults in `settings.py` and sample values in `config/config.yaml`.

**Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest -q tests/test_settings_orchestration.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add config/settings.py config/config.yaml tests/test_settings_orchestration.py
git commit -m "feat: add orchestration configuration"
```

### Task 4: Integrate Evidence-Aware Output With Existing Memory Context

**Files:**
- Modify: `core/agent_core.py`
- Modify: `core/session_manager.py` (if only metadata pass-through needed)
- Test: `tests/test_agent_evidence_integration.py`

**Step 1: Write the failing test**

Add tests for:
- memory-only question does not trigger search tools
- tool-using answer includes evidence bullets with tool name + query/snippet
- evidence block remains concise and deterministic

**Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest -q tests/test_agent_evidence_integration.py`
Expected: FAIL before integration behavior exists.

**Step 3: Write minimal implementation**

- ensure memory-based intents can complete without web-search tool path
- ensure tool calls are evidence-linked in final answer (`[Evidence]` section)
- keep compatibility with current channel stream/save flow

**Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest -q tests/test_agent_evidence_integration.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add core/agent_core.py core/session_manager.py tests/test_agent_evidence_integration.py
git commit -m "feat: integrate evidence-aware response flow"
```

### Task 5: End-to-End Verification and Docs

**Files:**
- Modify: `docs/reference/tui_question_to_answer_diagrams.md`
- Modify: `docs/plans/2026-02-13-agent-self-orchestration-refactor.md` (mark completion notes)

**Step 1: Write the failing verification checklist**

Create checklist covering:
- routing decisions
- tool usage reduction for memory/chat queries
- evidence section on tool-backed answers
- no regression in streaming + save flow

**Step 2: Run full targeted tests**

Run:
- `./.venv/Scripts/python -m pytest -q tests/test_agent_orchestration.py`
- `./.venv/Scripts/python -m pytest -q tests/test_agent_core_orchestration.py`
- `./.venv/Scripts/python -m pytest -q tests/test_agent_evidence_integration.py`
- `./.venv/Scripts/python -m pytest -q tests/test_session_manager.py -k "prepare_messages_for_agent or memory"`

Expected: all pass.

**Step 3: Update architecture docs**

Update `docs/reference/tui_question_to_answer_diagrams.md` with the new stages:
- Router
- Policy Scheduler
- Planner
- Executor
- Verifier/Evidence

**Step 4: Final sanity check**

Run: `./.venv/Scripts/python -m py_compile core/agent_core.py core/agent_orchestration.py config/settings.py`
Expected: no syntax errors.

**Step 5: Commit**

```bash
git add docs/reference/tui_question_to_answer_diagrams.md docs/plans/2026-02-13-agent-self-orchestration-refactor.md
git commit -m "docs: update runtime orchestration architecture"
```

## Non-Goals (YAGNI)

- No multi-agent delegation rewrite in this refactor
- No external vector DB migration
- No new tools; reuse existing tool registry
- No UI redesign

## Acceptance Criteria

- Agent uses intent-based self-scheduling before tool loop
- Tool usage is constrained by policy and intent
- Complex queries receive planned execution context
- Tool-backed conclusions are evidence-linked in final output
- Existing memory extraction pipeline keeps working unchanged
- Streaming event contract (`text/tool_call/tool_result/error/done`) remains compatible
