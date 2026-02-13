# TUI 提问到答案输出：时序图与流程图

本文梳理从用户在 TUI 提交问题，到最终在界面看到答案并落库的核心链路。

## 时序图（Sequence Diagram）

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant I as TUI 输入框(Input)
    participant A as ShiYiApp
    participant S as SessionManager
    participant DB as MemoryStorage
    participant C as AgentCore
    participant L as OpenAICompatibleEngine
    participant R as ToolRegistry
    participant T as Tool
    participant V as ChatView

    U->>I: 输入问题并回车
    I->>A: on_input_submitted(text)
    A->>A: _send_message() / _process_message()

    A->>S: save_message(session_id, "user", text)
    S->>DB: save_message()
    S->>DB: enqueue_embedding_job()
    opt 用户消息触发后台记忆提取
        S->>S: _fire_llm_extraction(content)
    end

    A->>V: add_user_message(text)
    A->>V: add_thinking()

    A->>S: get_session(current_session_id)
    S->>DB: get_messages()（或缓存命中）
    A->>S: prepare_messages_for_agent(messages)
    S->>DB: get_global_user_state()
    S->>S: 组装 memory_card / onboarding / recall 提示

    A->>C: process_message_stream(messages)
    C->>R: get_tool_definitions()
    C->>L: chat_with_tools_stream(full_messages, tools)

    loop 流式事件循环
        L-->>C: usage / text_delta / tool_calls
        alt text_delta
            C-->>A: {"type":"text","content":delta}
            A->>V: add/update assistant message
        else tool_calls
            C-->>A: {"type":"tool_call",...}
            A->>V: add_tool_call()
            C->>R: get_tool(name)
            R-->>C: tool instance
            C->>T: execute(args)
            T-->>C: result
            C-->>A: {"type":"tool_result","result":...}
            A->>V: update_tool_result()
            C->>L: 将 tool result 回填消息，再次推理
        else error
            C-->>A: {"type":"error",...}
            A->>V: add_error()
        end
    end

    C-->>A: {"type":"done"}
    opt full_response_text 非空
        A->>S: save_message(session_id, "assistant", full_response_text)
        S->>DB: save_message()
    end
    A->>V: 关闭 processing、恢复输入焦点
```

## 流程图（Flowchart）

```mermaid
flowchart TD
    START([用户在 TUI 输入并提交]) --> A{是否为空输入?}
    A -- 是 --> END0([忽略并返回])
    A -- 否 --> B{是否 Slash 命令?}

    B -- 是 --> C[执行 _handle_command]
    C --> END1([命令结果输出并结束])

    B -- 否 --> D{当前是否已有处理中回复?}
    D -- 是 --> E[提示“请等待当前回复完成”]
    E --> END2([结束])

    D -- 否 --> F[启动 _process_message worker]
    F --> G[set_processing = true]
    G --> H[保存用户消息: SessionManager.save_message]
    H --> I[ChatView 展示用户消息 + thinking]
    I --> J[获取会话上下文 + prepare_messages_for_agent]
    J --> K[调用 AgentCore.process_message_stream]

    K --> L{事件类型}
    L -- text --> M[累积 full_response_text 并更新助手气泡]
    M --> L
    L -- tool_call --> N[展示工具调用卡片]
    N --> L
    L -- tool_result --> O[更新工具结果/耗时]
    O --> L
    L -- error --> P[展示错误信息]
    P --> L
    L -- done --> Q{full_response_text 是否非空?}

    Q -- 是 --> R[保存助手回复到 SessionManager]
    R --> S[set_processing = false]
    Q -- 否 --> S

    S --> T[输入框重新聚焦]
    T --> END3([结束])
```

## 关键代码入口

- `channels/tui/app.py`：`on_input_submitted`、`_process_message`
- `core/session_manager.py`：`save_message`、`prepare_messages_for_agent`
- `core/agent_core.py`：`process_message_stream`
- `engines/llm/openai_compatible_engine.py`：`chat_with_tools_stream`
