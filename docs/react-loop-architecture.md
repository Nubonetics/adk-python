# ReAct Loop Architecture in Google ADK Python

This document provides a detailed technical analysis of the internal architecture of the
Reasoning-Action (ReAct) loop that powers AI agents in the Google Agent Development Kit
(ADK) for Python. It covers the full execution pipeline from the Runner entry point
through the LLM flow loop, tool dispatch, event propagation, and session persistence.

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [Agent Class Hierarchy](#2-agent-class-hierarchy)
3. [Runner and Session System](#3-runner-and-session-system)
4. [InvocationContext: The Central State Container](#4-invocationcontext-the-central-state-container)
5. [The Core ReAct Loop](#5-the-core-react-loop)
6. [Request Preprocessing Pipeline](#6-request-preprocessing-pipeline)
7. [LLM Invocation and Model Callbacks](#7-llm-invocation-and-model-callbacks)
8. [Postprocessing and Tool Dispatch](#8-postprocessing-and-tool-dispatch)
9. [Tool Execution System](#9-tool-execution-system)
10. [Event System and Propagation](#10-event-system-and-propagation)
11. [Agent Transfer and Multi-Agent Orchestration](#11-agent-transfer-and-multi-agent-orchestration)
12. [Callback Architecture](#12-callback-architecture)
13. [Flow Variants: SingleFlow vs AutoFlow](#13-flow-variants-singleflow-vs-autoflow)
14. [End-to-End Execution Example](#14-end-to-end-execution-example)
15. [Key File Reference](#15-key-file-reference)

---

## 1. Architectural Overview

The ADK ReAct loop implements a cyclic **Reasoning → Action → Observation** pattern:

1. **Reasoning**: The LLM receives conversation history and tool definitions, then
   produces a response that may include text, function calls, or both.
2. **Action**: If the LLM response contains function calls, the framework dispatches
   them to the appropriate tools, executing them in parallel where possible.
3. **Observation**: Tool results are packaged as function response events and appended
   to the session history, becoming visible to the LLM on the next iteration.
4. **Termination**: The loop exits when the LLM produces a final text response with no
   pending function calls.

The architecture is organized in layers:

```
┌────────────────────────────────────────────────────────────┐
│                     Caller / Application                    │
├────────────────────────────────────────────────────────────┤
│                     Runner (runners.py)                      │
│         Orchestration, session management, plugins           │
├────────────────────────────────────────────────────────────┤
│                 BaseAgent / LlmAgent                         │
│          Agent lifecycle, before/after callbacks              │
├────────────────────────────────────────────────────────────┤
│              BaseLlmFlow (SingleFlow / AutoFlow)             │
│        Core ReAct loop, request/response processors          │
├────────────────────────────────────────────────────────────┤
│                 Tool Execution (functions.py)                 │
│          Parallel dispatch, tool callbacks, auth             │
├────────────────────────────────────────────────────────────┤
│                  LLM Backend (BaseLlm)                       │
│           Model invocation, streaming, live API              │
└────────────────────────────────────────────────────────────┘
```

---

## 2. Agent Class Hierarchy

All agents inherit from `BaseAgent` (`src/google/adk/agents/base_agent.py`), which
defines the abstract interface for agent execution.

### BaseAgent

The abstract base class provides:

- **`run_async(parent_context)`** (final): The public entry point for text-based
  execution. Creates an invocation context, runs before/after agent callbacks, and
  delegates to `_run_async_impl()`.
- **`run_live(parent_context)`** (final): The public entry point for audio/video-based
  execution. Same lifecycle as `run_async` but delegates to `_run_live_impl()`.
- **`_run_async_impl(ctx)`** (abstract): Subclass-specific execution logic.
- **`_run_live_impl(ctx)`** (abstract): Subclass-specific live execution logic.
- **Parent/child composition**: `sub_agents` list with automatic parent reference
  wiring and name uniqueness validation.
- **Agent tree navigation**: `root_agent`, `find_agent(name)`, `find_sub_agent(name)`.

### LlmAgent (aliased as Agent)

`LlmAgent` (`src/google/adk/agents/llm_agent.py`) is the primary concrete agent that
integrates with an LLM through the flow system:

| Field | Purpose |
|---|---|
| `model` | LLM model name or instance. Inherited from parent if empty. |
| `instruction` | Dynamic instruction string or callable. Supports state injection. |
| `static_instruction` | Fixed instruction for context caching. |
| `tools` | List of tools (functions, `BaseTool`, `BaseToolset`). |
| `generate_content_config` | Temperature, safety settings, etc. |
| `planner` | Optional planning module for step-by-step execution. |
| `code_executor` | Optional code execution from model responses. |
| `output_schema` | Pydantic model for structured output. |
| `output_key` | Session state key to store agent output. |
| `include_contents` | `'default'` (full history) or `'none'` (current turn only). |
| `disallow_transfer_to_parent` | Prevents upward agent transfer. |
| `disallow_transfer_to_peers` | Prevents sibling agent transfer. |

**Model resolution chain**: self → ancestor LlmAgents → class default (`gemini-2.5-flash`).

**Flow selection** (`_llm_flow` property):
- If the agent has no sub-agents and transfer is disabled → `SingleFlow`
- Otherwise → `AutoFlow` (adds agent transfer capability)

### Shell Agent Types

These agents compose and coordinate sub-agents without LLM capability:

| Agent | Behavior |
|---|---|
| `SequentialAgent` | Runs sub-agents one after another in order. |
| `ParallelAgent` | Runs sub-agents concurrently with branch isolation via `asyncio.TaskGroup`. |
| `LoopAgent` | Repeats sub-agents in a loop until `max_iterations` or escalation. |

### Specialized Agents

| Agent | Behavior |
|---|---|
| `LangGraphAgent` | Integrates LangGraph compiled graphs, converting between ADK events and LangChain messages. |
| `RemoteA2aAgent` | Communicates with remote agents over the A2A (Agent-to-Agent) protocol via HTTP. |

### Class Hierarchy Diagram

```
BaseAgent [Abstract]
├── LlmAgent (alias: Agent)
│   └── Uses BaseLlmFlow (SingleFlow or AutoFlow)
├── SequentialAgent
├── ParallelAgent
├── LoopAgent
├── LangGraphAgent
└── RemoteA2aAgent
```

---

## 3. Runner and Session System

### Runner

The `Runner` class (`src/google/adk/runners.py`) is the top-level orchestrator.

**Key responsibilities**:
- Create and manage `InvocationContext` for each execution
- Retrieve or auto-create sessions via `SessionService`
- Wrap execution with plugin lifecycle callbacks (`before_run`, `on_event`, `after_run`)
- Persist events to session storage
- Support invocation resumption for long-running tools

**Entry point — `run_async()`**:

```python
async def run_async(
    self, *, user_id, session_id,
    invocation_id=None,  # for resumption
    new_message=None,
    state_delta=None,
    run_config=None,
) -> AsyncGenerator[Event, None]:
```

Execution stages:

1. **Session retrieval**: Load or create session from `SessionService`.
2. **Context setup**: Build `InvocationContext` (new or resumed).
3. **Agent discovery**: Determine which agent handles the message via
   `_find_agent_to_run()`, which examines session history for function call
   continuity or recent agent activity.
4. **Plugin-wrapped execution**: Run `agent.run_async(ctx)` inside
   `_exec_with_plugin()`, which manages `before_run_callback`, `on_event_callback`,
   and `after_run_callback`.
5. **Event persistence**: Each non-partial event is appended to the session via
   `session_service.append_event()`.
6. **Post-invocation compaction**: Optionally compact old events if configured.

### Session

The `Session` class (`src/google/adk/sessions/session.py`) holds conversation state:

```python
class Session:
    id: str
    app_name: str
    user_id: str
    state: dict[str, Any]  # Mutable session state
    events: list[Event]     # Conversation event history
    last_update_time: float
```

**State partitioning** uses key prefixes:

| Prefix | Scope | Persistence |
|---|---|---|
| `app:` | Shared across all users | Persisted at app level |
| `user:` | Shared across user's sessions | Persisted at user level |
| `temp:` | Temporary | Never persisted |
| *(no prefix)* | Session-local | Persisted with session |

The `State` wrapper class (`src/google/adk/sessions/state.py`) tracks a `value` dict
(committed state) and a `delta` dict (uncommitted changes), enabling transactional
state updates that are written to `EventActions.state_delta`.

---

## 4. InvocationContext: The Central State Container

`InvocationContext` (`src/google/adk/agents/invocation_context.py`) is created per
invocation and threaded through the entire execution pipeline:

| Category | Fields |
|---|---|
| **Identity** | `invocation_id`, `branch`, `agent`, `user_content` |
| **Session** | `session`, `session_service` |
| **Services** | `artifact_service`, `memory_service`, `credential_service` |
| **Agent state** | `agent_states: dict[str, dict]`, `end_of_agents: dict[str, bool]` |
| **Control** | `end_invocation: bool`, `run_config`, `resumability_config` |
| **Plugins** | `plugin_manager` |
| **Streaming** | `live_request_queue`, `active_streaming_tools`, transcription caches |
| **Cost tracking** | `_invocation_cost_manager` (enforces `max_llm_calls`) |

Key methods:
- `set_agent_state()` — Save or clear agent checkpoint state for resumability.
- `populate_invocation_agent_states()` — Restore states from session history.
- `increment_llm_call_count()` — Track and enforce LLM call limits.
- `_get_events(current_invocation, current_branch)` — Filter session events by
  invocation or branch for context isolation.
- `should_pause_invocation(event)` — Check if a long-running tool requires pausing.

**Context hierarchy for callbacks**:
- `ReadonlyContext` — Read-only view (used in guardrails and read-only hooks).
- `CallbackContext` (extends `ReadonlyContext`) — Mutable state access, artifact and
  credential operations. Used in before/after callbacks.
- `ToolContext` (extends `CallbackContext`) — Additional tool-specific operations:
  `request_credential()`, `request_confirmation()`, `search_memory()`.

---

## 5. The Core ReAct Loop

The loop lives in `BaseLlmFlow.run_async()` (`src/google/adk/flows/llm_flows/base_llm_flow.py`):

```python
async def run_async(self, invocation_context):
    while True:
        last_event = None
        async for event in self._run_one_step_async(invocation_context):
            last_event = event
            yield event

        if not last_event or last_event.is_final_response() or last_event.partial:
            break
```

**Loop termination conditions**:
1. No events produced (`last_event is None`)
2. Final response: no function calls, no function responses, not partial
3. Partial response (unexpected, logged as warning)

Each iteration calls `_run_one_step_async()`, which performs one complete
**Preprocess → LLM Call → Postprocess** cycle:

```
┌──────────────────────── _run_one_step_async ─────────────────────────┐
│                                                                       │
│  1. _preprocess_async(invocation_context, llm_request)               │
│     ├─ Run all request processors in order                           │
│     ├─ Resolve toolset authentication                                │
│     └─ Process and append tool declarations                          │
│                                                                       │
│  2. Resume check (for long-running tools)                            │
│     └─ If resuming and last event has pending function calls,        │
│        execute them directly without calling the LLM                 │
│                                                                       │
│  3. _call_llm_async(invocation_context, llm_request, event)         │
│     ├─ before_model_callback                                         │
│     ├─ llm.generate_content_async()                                  │
│     └─ after_model_callback                                          │
│                                                                       │
│  4. _postprocess_async(invocation_context, llm_request, response)    │
│     ├─ Run response processors                                       │
│     ├─ Yield model response event                                    │
│     └─ If function calls present:                                    │
│        _postprocess_handle_function_calls_async()                    │
│        ├─ Execute tools (parallel)                                   │
│        ├─ Yield auth/confirmation events if needed                   │
│        ├─ Yield function response event                              │
│        └─ Handle agent transfer if triggered                         │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

The loop continues because the function response event is not a final response (it
contains function responses), causing the next iteration to include the tool results
in the conversation history sent to the LLM.

---

## 6. Request Preprocessing Pipeline

Preprocessing runs all registered `BaseLlmRequestProcessor` instances in order,
each modifying the `LlmRequest` in place.

### LlmRequest Structure

`LlmRequest` (`src/google/adk/models/llm_request.py`) accumulates everything needed
for one LLM call:

| Field | Purpose |
|---|---|
| `model` | Model name |
| `contents` | Conversation history (list of `Content`) |
| `config` | `GenerateContentConfig` with system instruction, tool declarations, etc. |
| `tools_dict` | Map of tool name → `BaseTool` for dispatch |
| `cache_config` | Context caching configuration |
| `previous_interaction_id` | For stateful Interactions API |

Key methods:
- `append_instructions(instructions)` — Append to system instruction text.
- `append_tools(tools)` — Add tool declarations to the config.
- `set_output_schema(base_model)` — Set structured output JSON schema.

### Processor Chain (SingleFlow)

`SingleFlow` (`src/google/adk/flows/llm_flows/single_flow.py`) registers these
request processors in order:

| # | Processor | File | Purpose |
|---|---|---|---|
| 1 | `basic` | `basic.py` | Sets model name, merge `generate_content_config` |
| 2 | `auth_preprocessor` | `auth_preprocessor.py` | Handles tool authentication setup |
| 3 | `request_confirmation` | `request_confirmation.py` | Processes pending tool confirmations |
| 4 | `instructions` | `instructions.py` | Composes system instructions with state injection |
| 5 | `identity` | `identity.py` | Injects agent name and description as instruction |
| 6 | `contents` | `contents.py` | Builds filtered conversation history |
| 7 | `context_cache_processor` | `context_cache_processor.py` | Context caching metadata |
| 8 | `interactions_processor` | `interactions_processor.py` | Stateful conversation chaining |
| 9 | `_nl_planning` | `_nl_planning.py` | Natural language planning setup |
| 10 | `_code_execution` | `_code_execution.py` | Code executor tool registration |
| 11 | `_output_schema_processor` | `_output_schema_processor.py` | Output schema enforcement |

`AutoFlow` adds one more:

| 12 | `agent_transfer` | `agent_transfer.py` | Adds `transfer_to_agent` tool and instructions |

### Key Processors in Detail

**Instructions processor** (`instructions.py`):
- Resolves `global_instruction` (deprecated), `static_instruction`, and `instruction`.
- Static instructions go into system instruction (cacheable).
- Dynamic instructions go into user content if static instructions exist, or into
  system instruction otherwise.
- Supports session state injection into instruction templates.

**Contents processor** (`contents.py`):
- Filters session events by branch for agent isolation.
- Processes event compaction summaries.
- Aggregates consecutive transcriptions.
- Converts other agents' messages to user-role content with `[agent_name] said:` prefix.
- Rearranges function calls and responses to be consecutive.
- Strips internal ADK framework events (auth, confirmations).

**Agent transfer processor** (`agent_transfer.py`):
- Discovers available transfer targets: sub-agents, parent (if allowed), peers (if
  allowed).
- Creates a `TransferToAgentTool` with available agent names.
- Injects transfer instructions describing each agent's capabilities.

---

## 7. LLM Invocation and Model Callbacks

The LLM call is managed by `_call_llm_async()` in `BaseLlmFlow`:

```
_call_llm_async(invocation_context, llm_request, model_response_event)
    │
    ├─ _handle_before_model_callback()
    │   ├─ Plugin callbacks (highest priority)
    │   └─ Agent canonical_before_model_callbacks
    │   → If callback returns LlmResponse, skip actual LLM call
    │
    ├─ llm.generate_content_async(llm_request, stream=True)
    │   → Yields LlmResponse chunks (streaming)
    │
    ├─ _handle_after_model_callback()
    │   ├─ Plugin callbacks (highest priority)
    │   └─ Agent canonical_after_model_callbacks
    │   → Can replace the LLM response
    │
    └─ On error: _run_on_model_error_callbacks()
        ├─ Plugin callbacks
        └─ Agent canonical on_model_error_callbacks
        → Can provide fallback response
```

The `before_model_callback` can short-circuit the LLM call entirely by returning
an `LlmResponse`, enabling patterns like response caching, mocking, or guardrails.

---

## 8. Postprocessing and Tool Dispatch

After the LLM responds, `_postprocess_async()` handles the response:

1. **Response processors**: Run registered `BaseLlmResponseProcessor` instances
   (planning processor, code execution processor).
2. **Event finalization**: Merge `LlmResponse` into an `Event`, populate function
   call IDs, identify long-running tools.
3. **Function call handling**: If the event contains function calls,
   `_postprocess_handle_function_calls_async()` takes over:

```python
async def _postprocess_handle_function_calls_async(self, ...):
    # Execute all tool calls
    function_response_event = await functions.handle_function_calls_async(
        invocation_context, function_call_event, llm_request.tools_dict
    )

    # Generate auth event if tools need authentication
    auth_event = functions.generate_auth_event(invocation_context, function_response_event)
    if auth_event:
        yield auth_event

    # Generate confirmation event if tools need user approval
    tool_confirmation_event = functions.generate_request_confirmation_event(...)
    if tool_confirmation_event:
        yield tool_confirmation_event

    # Yield the tool response (fed back to LLM on next iteration)
    yield function_response_event

    # Handle structured output from output_schema tools
    if json_response := _output_schema_processor.get_structured_model_response(...):
        yield _output_schema_processor.create_final_model_response_event(...)

    # Handle agent transfer
    transfer_to_agent = function_response_event.actions.transfer_to_agent
    if transfer_to_agent:
        agent_to_run = self._get_agent_to_run(invocation_context, transfer_to_agent)
        async for event in agent_to_run.run_async(invocation_context):
            yield event
```

The function response event becomes part of the session history. On the next loop
iteration, the contents processor includes it in `llm_request.contents`, closing
the feedback cycle.

---

## 9. Tool Execution System

### Tool Class Hierarchy

```
BaseTool [Abstract]
├── FunctionTool (wraps Python callables)
│   ├── AuthenticatedFunctionTool (adds credential injection)
│   └── TransferToAgentTool (special: triggers agent transfer)
├── BaseAuthenticatedTool (abstract, for custom auth tools)
└── Various built-in tools (GoogleSearchTool, VertexAiSearchTool, etc.)
```

`BaseToolset` provides collections of tools with filtering and prefix support.

### Dispatch Pipeline

Tool execution is orchestrated by `functions.py` (`src/google/adk/flows/llm_flows/functions.py`):

**Parallel execution**: When the LLM produces multiple function calls in one response,
they are executed concurrently:

```python
async def handle_function_call_list_async(...):
    tasks = [
        asyncio.create_task(
            _execute_single_function_call_async(
                invocation_context, function_call, tools_dict, agent, ...
            )
        )
        for function_call in filtered_calls
    ]
    function_response_events = await asyncio.gather(*tasks)
    merged_event = merge_parallel_function_response_events(function_response_events)
    return merged_event
```

### Single Tool Execution Pipeline

`_execute_single_function_call_async()` implements a six-step pipeline:

```
Step 1: Plugin before_tool_callback
        → If returns non-None dict, use as result (skip tool)

Step 2: Agent canonical_before_tool_callbacks
        → Iterate until one returns non-None (skip tool)

Step 3: Execute tool.run_async(args, tool_context)
        → On exception: run on_tool_error_callbacks
          → If callback returns dict, use as result
          → Otherwise, re-raise exception

Step 4: Plugin after_tool_callback
        → Can replace the tool response

Step 5: Agent canonical_after_tool_callbacks
        → Can replace the tool response

Step 6: Build function response Event
        → Wrap result as FunctionResponse part
        → Include EventActions (state_delta, auth requests, etc.)
```

### FunctionTool Execution Details

`FunctionTool` (`src/google/adk/tools/function_tool.py`) wraps Python callables:

1. **Parameter validation**: Checks all mandatory parameters are present, returns
   error dict if missing (allowing the LLM to retry).
2. **Argument preprocessing**: Converts JSON dicts to Pydantic models based on type hints.
3. **Confirmation flow**: If `require_confirmation=True`, requests user approval
   before execution.
4. **Invocation**: Handles both sync and async callables, injecting `tool_context`
   if the function signature accepts it.

### Tool Result Construction

Tool results are packaged by `__build_response_event()`:

```python
def __build_response_event(tool, function_result, tool_context, invocation_context):
    if not isinstance(function_result, dict):
        function_result = {'result': function_result}

    part = types.Part.from_function_response(
        name=tool.name, response=function_result
    )
    part.function_response.id = tool_context.function_call_id

    return Event(
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
        content=types.Content(role='user', parts=[part]),
        actions=tool_context.actions,
        branch=invocation_context.branch,
    )
```

When multiple tools execute in parallel, `merge_parallel_function_response_events()`
deep-merges all `EventActions` and combines all parts into a single event.

### Authentication Flow

For tools requiring authentication:

1. Tool calls `tool_context.request_credential(auth_config)`.
2. Framework generates an `adk_request_credential` function call event.
3. User provides auth response externally.
4. ADK exchanges and caches the credential.
5. Tool is retried with the credential available via `tool_context.get_auth_response()`.

### Long-Running Tools

Tools with `is_long_running=True`:
- Are identified during event finalization.
- Cause `Event.long_running_tool_ids` to be set.
- Trigger invocation pausing via `should_pause_invocation()`.
- Can be resumed via `Runner.run_async(invocation_id=...)`.

---

## 10. Event System and Propagation

### Event Structure

`Event` (`src/google/adk/events/event.py`) extends `LlmResponse`:

| Field | Purpose |
|---|---|
| `invocation_id` | Links event to its invocation |
| `author` | `'user'` or agent name |
| `content` | Model text, function calls, or function responses |
| `actions` | `EventActions` with metadata (see below) |
| `long_running_tool_ids` | IDs of long-running tools requiring pause |
| `branch` | Agent hierarchy branch for context isolation |
| `id` | UUID for uniqueness |
| `timestamp` | Creation time |
| `partial` | Whether this is a streaming partial event |

### EventActions

`EventActions` (`src/google/adk/events/event_actions.py`) carries side effects:

| Field | Purpose |
|---|---|
| `state_delta` | Session state updates |
| `artifact_delta` | Artifact version tracking |
| `transfer_to_agent` | Target agent for transfer |
| `escalate` | Escalation flag (exits LoopAgent) |
| `skip_summarization` | Skip LLM summarization of this event |
| `requested_auth_configs` | Pending authentication requests |
| `requested_tool_confirmations` | Pending tool confirmations |
| `end_of_agent` | Agent completion marker |
| `agent_state` | Checkpoint state for resumability |

### Event Flow Through the System

```
Agent generates Event
        ↓
Runner._exec_with_plugin() receives event
        ↓
Apply run_config custom metadata
        ↓
Append to session (non-partial events)
        ↓
Run plugin_manager.run_on_event_callback()
  → May modify event
        ↓
Yield to caller (API response / stream)
```

### `is_final_response()` Logic

An event is considered a final response when:
- It has no function calls
- It has no function responses
- It is not marked `partial`
- It has no trailing code execution results
- OR it has `skip_summarization` or `long_running_tool_ids` set

---

## 11. Agent Transfer and Multi-Agent Orchestration

### Transfer Mechanism

When `AutoFlow` is active, the agent transfer processor adds a `transfer_to_agent`
tool to the LLM request. If the LLM calls this tool:

1. The tool sets `event_actions.transfer_to_agent = target_agent_name`.
2. In `_postprocess_handle_function_calls_async()`, the framework detects this action.
3. It resolves the target agent via `_get_agent_to_run()`.
4. It invokes `target_agent.run_async(invocation_context)`, yielding all resulting
   events.

### Transfer Targets

Available transfer targets depend on agent configuration:

| Target | Condition |
|---|---|
| Sub-agents | Always available |
| Parent agent | Unless `disallow_transfer_to_parent=True` |
| Peer agents (siblings) | Unless `disallow_transfer_to_peers=True` |

### Branch Isolation

Each agent in a hierarchy operates on a branch identified by
`agent_1.agent_2.agent_3`. The contents processor filters events by branch,
ensuring agents only see their own conversation context.

---

## 12. Callback Architecture

The ADK provides callbacks at three levels, each following a consistent pattern:
**plugins execute first** (highest priority), then **agent-level callbacks**. The
first non-`None` return wins.

### Agent-Level Callbacks

| Callback | When | Can Override |
|---|---|---|
| `before_agent_callback` | Before `_run_async_impl()` | Return content to skip agent entirely |
| `after_agent_callback` | After `_run_async_impl()` | Append additional content |

### Model-Level Callbacks

| Callback | When | Can Override |
|---|---|---|
| `before_model_callback` | Before LLM call | Return `LlmResponse` to skip LLM call |
| `after_model_callback` | After LLM response | Replace the LLM response |
| `on_model_error_callback` | On LLM error | Provide fallback response |

### Tool-Level Callbacks

| Callback | When | Can Override |
|---|---|---|
| `before_tool_callback` | Before tool execution | Return dict to skip tool, use as result |
| `after_tool_callback` | After tool execution | Replace the tool result |
| `on_tool_error_callback` | On tool exception | Provide fallback result |

### Callback Signatures

```python
# Agent callbacks
BeforeAgentCallback = Callable[[CallbackContext], Optional[types.Content]]
AfterAgentCallback  = Callable[[CallbackContext], Optional[types.Content]]

# Model callbacks
BeforeModelCallback   = Callable[[CallbackContext, LlmRequest], Optional[LlmResponse]]
AfterModelCallback    = Callable[[CallbackContext, LlmResponse], Optional[LlmResponse]]
OnModelErrorCallback  = Callable[[CallbackContext, LlmRequest, Exception], Optional[LlmResponse]]

# Tool callbacks
BeforeToolCallback   = Callable[[BaseTool, dict, ToolContext], Optional[dict]]
AfterToolCallback    = Callable[[BaseTool, dict, ToolContext, dict], Optional[dict]]
OnToolErrorCallback  = Callable[[BaseTool, dict, ToolContext, Exception], Optional[dict]]
```

All callbacks support both sync and async variants (detected and awaited automatically).

---

## 13. Flow Variants: SingleFlow vs AutoFlow

### SingleFlow

Basic ReAct loop with tools but no agent transfer:

- 11 request processors (basic, auth, confirmation, instructions, identity, contents,
  cache, interactions, planning, code execution, output schema)
- 2 response processors (planning, code execution)
- Selected when: `disallow_transfer_to_parent=True` AND `disallow_transfer_to_peers=True`
  AND no sub-agents

### AutoFlow

Extends `SingleFlow` with agent transfer:

- Adds `agent_transfer` request processor
- Enables `transfer_to_agent` tool in LLM requests
- Handles agent delegation in postprocessing
- Selected when: agent has sub-agents OR transfer is allowed

---

## 14. End-to-End Execution Example

Consider a user asking: *"What's the weather in New York?"*

```
1. Runner.run_async(user_id, session_id, new_message="What's the weather?")
   ├─ Load session
   ├─ Create InvocationContext
   └─ Append user message to session

2. BaseAgent.run_async(ctx)
   ├─ before_agent_callback → None (no override)
   └─ LlmAgent._run_async_impl(ctx)
       └─ self._llm_flow.run_async(ctx)

3. BaseLlmFlow.run_async(ctx) — LOOP ITERATION 1
   └─ _run_one_step_async(ctx)

       3a. _preprocess_async():
           ├─ Instructions: "You are a helpful assistant..."
           ├─ Identity: 'You are agent "weather_agent"'
           ├─ Contents: [user: "What's the weather in New York?"]
           └─ Tools: [get_weather(location: str)]

       3b. _call_llm_async():
           └─ LLM responds: FunctionCall(name="get_weather", args={"location": "New York"})

       3c. _postprocess_async():
           ├─ Yield Event(function_call: get_weather)
           └─ _postprocess_handle_function_calls_async():
               ├─ Execute get_weather("New York") → {"temp": "72°F", "condition": "sunny"}
               └─ Yield Event(function_response: {temp: "72°F", condition: "sunny"})

   └─ last_event has function_response → not is_final_response() → CONTINUE

4. BaseLlmFlow.run_async(ctx) — LOOP ITERATION 2
   └─ _run_one_step_async(ctx)

       4a. _preprocess_async():
           └─ Contents now includes:
              [user: "What's the weather?"]
              [model: FunctionCall(get_weather)]
              [user: FunctionResponse({temp: "72°F", condition: "sunny"})]

       4b. _call_llm_async():
           └─ LLM responds: "The weather in New York is 72°F and sunny."

       4c. _postprocess_async():
           └─ Yield Event(text: "The weather in New York is 72°F and sunny.")

   └─ last_event.is_final_response() → True → BREAK

5. BaseAgent.run_async(ctx)
   └─ after_agent_callback → None

6. Runner._exec_with_plugin():
   ├─ Append events to session
   ├─ Run on_event_callback for each event
   └─ Yield events to caller
```

---

## 15. Key File Reference

| File | Purpose |
|---|---|
| `src/google/adk/runners.py` | Runner: top-level orchestration, session management, plugin wrapping |
| `src/google/adk/agents/base_agent.py` | BaseAgent: abstract agent interface, before/after callbacks |
| `src/google/adk/agents/llm_agent.py` | LlmAgent: LLM-powered agent, tool/model configuration, flow selection |
| `src/google/adk/agents/invocation_context.py` | InvocationContext: per-invocation state container |
| `src/google/adk/agents/callback_context.py` | CallbackContext: mutable context for callbacks |
| `src/google/adk/agents/readonly_context.py` | ReadonlyContext: read-only context view |
| `src/google/adk/flows/llm_flows/base_llm_flow.py` | BaseLlmFlow: core ReAct loop, LLM call, pre/post processing |
| `src/google/adk/flows/llm_flows/single_flow.py` | SingleFlow: basic tool-calling flow with processor registration |
| `src/google/adk/flows/llm_flows/auto_flow.py` | AutoFlow: extends SingleFlow with agent transfer |
| `src/google/adk/flows/llm_flows/functions.py` | Tool dispatch: parallel execution, callbacks, response merging |
| `src/google/adk/flows/llm_flows/contents.py` | Contents processor: conversation history construction |
| `src/google/adk/flows/llm_flows/instructions.py` | Instructions processor: system instruction composition |
| `src/google/adk/flows/llm_flows/agent_transfer.py` | Agent transfer processor: sub-agent delegation |
| `src/google/adk/flows/llm_flows/_base_llm_processor.py` | Processor base classes |
| `src/google/adk/models/llm_request.py` | LlmRequest: data structure for model requests |
| `src/google/adk/models/llm_response.py` | LlmResponse: data structure for model responses |
| `src/google/adk/events/event.py` | Event: immutable communication unit |
| `src/google/adk/events/event_actions.py` | EventActions: side-effect metadata |
| `src/google/adk/tools/base_tool.py` | BaseTool: abstract tool interface |
| `src/google/adk/tools/function_tool.py` | FunctionTool: wraps Python callables as tools |
| `src/google/adk/tools/tool_context.py` | ToolContext: per-tool-call context |
| `src/google/adk/tools/base_toolset.py` | BaseToolset: tool collection abstraction |
| `src/google/adk/sessions/session.py` | Session: conversation state container |
| `src/google/adk/sessions/base_session_service.py` | BaseSessionService: session persistence interface |
| `src/google/adk/sessions/state.py` | State: transactional state wrapper with delta tracking |
| `src/google/adk/agents/sequential_agent.py` | SequentialAgent: runs sub-agents in order |
| `src/google/adk/agents/parallel_agent.py` | ParallelAgent: runs sub-agents concurrently |
| `src/google/adk/agents/loop_agent.py` | LoopAgent: repeats sub-agents in a loop |
