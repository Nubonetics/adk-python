# Google ADK (Agent Development Kit) — Implementation-Level Specification

> **Purpose**: This document specifies the exact runtime behavior of the Google ADK
> Python framework at a level of detail sufficient to generate behaviorally-identical
> source code from a YAML agent description. Every algorithm, predicate, data
> transformation, and edge case is documented with source file references.

---

## 1. Entry Point & Invocation Lifecycle

### 1.1 Public API Surface

The user-facing entry point is `Runner.run_async()` in `runners.py:453`.

```
Signature:
  run_async(
    user_id: str,
    session_id: str,
    invocation_id: Optional[str] = None,
    new_message: Optional[types.Content] = None,
    state_delta: Optional[dict[str, Any]] = None,
    run_config: Optional[RunConfig] = None,
  ) -> AsyncGenerator[Event, None]
```

**Execution sequence** (`runners.py:492–563`):

```
1. run_config = run_config or RunConfig()
2. IF new_message AND NOT new_message.role → set role = 'user'
3. Open tracing span 'invocation'
4. session = await _get_or_create_session(user_id, session_id)
5. IF invocation_id provided (resumption):
     a. Validate app is resumable
     b. invocation_context = _setup_context_for_resumed_invocation(...)
     c. IF invocation_context.end_of_agents[agent.name] → return immediately
   ELSE (new invocation):
     a. invocation_context = _setup_context_for_new_invocation(...)
6. Wrap agent execution with _exec_with_plugin()
7. After all events yielded, run compaction if configured
```

### 1.2 New Invocation Setup

`_setup_context_for_new_invocation()` (`runners.py:1256–1292`):

```
1. Create InvocationContext via _new_invocation_context(session, new_message, run_config)
2. Handle new message:
   a. Run plugin_manager.run_on_user_message_callback() → may modify message
   b. Append message to session as Event(author='user', content=new_message)
      - If state_delta provided, attach it to the user event
      - If message is a function_response, find matching function_call and copy its branch
3. Set agent: invocation_context.agent = _find_agent_to_run(session, root_agent)
```

### 1.3 InvocationContext Construction

`_new_invocation_context()` (`runners.py:1372–1420`):

```python
InvocationContext(
    artifact_service=self.artifact_service,
    session_service=self.session_service,
    memory_service=self.memory_service,
    credential_service=self.credential_service,
    plugin_manager=self.plugin_manager,
    context_cache_config=self.context_cache_config,
    invocation_id=invocation_id or new_invocation_context_id(),  # "e-" + uuid4()
    agent=self.agent,               # root agent initially
    session=session,
    user_content=new_message,
    live_request_queue=live_request_queue,
    run_config=run_config,
    resumability_config=self.resumability_config,
)
```

Key `InvocationContext` fields (`invocation_context.py`):

| Field | Type | Purpose |
|---|---|---|
| `invocation_id` | `str` | Unique per invocation, format `"e-{uuid4()}"` |
| `branch` | `Optional[str]` | Dot-delimited agent path for context isolation |
| `agent` | `BaseAgent` | Currently executing agent |
| `session` | `Session` | Session with events list |
| `end_invocation` | `bool` | Flag to halt the invocation |
| `is_resumable` | `bool` | Derived from `resumability_config` |
| `agent_states` | `dict[str, BaseAgentState]` | Per-agent state for checkpointing |
| `end_of_agents` | `dict[str, bool]` | Tracks which agents have finished |
| `_invocation_cost_manager` | Internal | Tracks LLM call count |

### 1.4 Agent Selection Algorithm

`_find_agent_to_run()` (`runners.py:1078–1131`):

```
1. Check if last session event is a function_response:
   - Find matching function_call event via find_matching_function_call()
   - If found, return root_agent.find_agent(event.author)

2. Iterate session.events in REVERSE, filtering out:
   - Events where author == 'user'
   - Events where actions.agent_state is not None or actions.end_of_agent

3. For each qualifying event:
   a. IF event.author == root_agent.name → return root_agent
   b. Find agent = root_agent.find_sub_agent(event.author)
   c. IF agent not found → skip (log warning)
   d. IF _is_transferable_across_agent_tree(agent) → return agent

4. Fallback: return root_agent
```

`_is_transferable_across_agent_tree()` (`runners.py:1132–1152`):
```
Walk from agent to root via parent_agent:
  - IF any ancestor is NOT LlmAgent → return False
  - IF any ancestor has disallow_transfer_to_parent → return False
Return True only if all ancestors allow upward transfer
```

### 1.5 Event Persistence in Runner

`_exec_with_plugin()` (`runners.py:720–851`) wraps every agent execution:

```
1. Run plugin_manager.run_before_run_callback()
   - If returns types.Content → yield early-exit event, skip agent

2. For each event from agent execution:
   a. Apply custom_metadata from run_config
   b. IF event.partial is NOT True:
        append_event(session, event)
   c. Run plugin_manager.run_on_event_callback() → may modify event
   d. Yield event (original or modified)

3. After all events: run plugin_manager.run_after_run_callback()
```

**Critical rule**: Events where `event.partial is True` are **never** persisted to the session.

---

## 2. The Core Loop (ReAct Pattern)

### 2.1 Agent Entry Point

`BaseAgent.run_async()` (`base_agent.py:270–301`):

```
1. Open tracing span 'invoke_agent {name}'
2. Create child InvocationContext via _create_invocation_context(parent_context)
3. Run before_agent_callback → if returns Content, yield event and stop
4. IF ctx.end_invocation → return
5. Yield all events from _run_async_impl(ctx)
6. IF ctx.end_invocation → return
7. Run after_agent_callback → if returns Content, yield extra event
```

`LlmAgent._run_async_impl()` (`llm_agent.py:448–485`):

```
1. agent_state = _load_agent_state(ctx, BaseAgentState)
2. IF agent_state is not None AND sub-agent needs resuming:
     - Run sub-agent, yield its events
     - Set end_of_agent = True, yield state event
     - RETURN

3. should_pause = False
4. FOR EACH event FROM self._llm_flow.run_async(ctx):
     a. __maybe_save_output_to_state(event)
     b. YIELD event
     c. IF ctx.should_pause_invocation(event):
          should_pause = True
5. IF should_pause → RETURN

6. IF ctx.is_resumable:
     - Check last 2 events for pause conditions
     - If no pause needed: set end_of_agent, yield state event
```

### 2.2 Flow Selection

`LlmAgent._llm_flow` property (`llm_agent.py:694–703`):

```python
IF (disallow_transfer_to_parent
    AND disallow_transfer_to_peers
    AND NOT sub_agents):
  return SingleFlow()
ELSE:
  return AutoFlow()
```

- **SingleFlow**: Agent + tools only, no transfers
- **AutoFlow**: Extends SingleFlow with `agent_transfer.request_processor`

### 2.3 Outer Loop — `BaseLlmFlow.run_async()`

`base_llm_flow.py:420–433`:

```python
async def run_async(self, invocation_context):
    while True:
        last_event = None
        for event in _run_one_step_async(invocation_context):
            last_event = event
            yield event
        # TERMINATION CONDITIONS (ALL must be checked):
        if not last_event:
            break
        if last_event.is_final_response():
            break
        if last_event.partial:
            logger.warning('The last event is partial, which is not expected.')
            break
```

**The loop continues** when `last_event.is_final_response()` returns `False` — meaning tool calls need to be executed and their results sent back to the LLM.

### 2.4 Termination Predicate — `Event.is_final_response()`

`event.py:82–97`:

```python
def is_final_response(self) -> bool:
    if self.actions.skip_summarization or self.long_running_tool_ids:
        return True
    return (
        not self.get_function_calls()        # No pending tool calls
        and not self.get_function_responses() # Not a tool response
        and not self.partial                  # Not a streaming fragment
        and not self.has_trailing_code_execution_result()  # No code output
    )
```

A response is **final** when:
- `skip_summarization=True` (tool set this flag), OR
- `long_running_tool_ids` is non-empty (async tools), OR
- The event has NO function_calls, NO function_responses, is NOT partial, and does NOT end with a code_execution_result

### 2.5 Per-Step Execution — `_run_one_step_async()`

`base_llm_flow.py:435–518`:

```
1. CREATE empty LlmRequest

2. PREPROCESS: run _preprocess_async(ctx, llm_request)
   - Run all request_processors in order
   - Resolve toolset auth
   - Process each tool via tool.process_llm_request()
   - IF ctx.end_invocation → RETURN

3. RESUME CHECK (for resumable invocations):
   - Get events from current invocation + branch
   - IF last 2 events should pause → RETURN
   - IF last event has function_calls:
       Execute them and yield responses, RETURN

4. LLM CALL:
   a. Create model_response_event skeleton:
      Event(id=new_id(), invocation_id=..., author=agent.name, branch=...)
   b. Call _call_llm_async(ctx, llm_request, model_response_event)
   c. For each llm_response:
      - Run _postprocess_async(ctx, llm_request, llm_response, model_response_event)
      - Yield each resulting event
      - Update model_response_event.id to new UUID (avoid conflicts)
```

---

## 3. Preprocessing Pipeline (Request Processors)

### 3.1 Processor Chain — SingleFlow

`single_flow.py:44–72`:

```python
request_processors = [
    basic.request_processor,              # 1. Model, config, output_schema
    auth_preprocessor.request_processor,  # 2. Auth credential resolution
    request_confirmation.request_processor, # 3. Tool confirmation handling
    instructions.request_processor,       # 4. System instructions
    identity.request_processor,           # 5. Agent identity
    contents.request_processor,           # 6. Conversation history
    context_cache_processor.request_processor,  # 7. Context caching
    interactions_processor.request_processor,   # 8. Interactions API
    _nl_planning.request_processor,       # 9. NL planning
    _code_execution.request_processor,    # 10. Code execution
    _output_schema_processor.request_processor, # 11. Output schema workaround
]

response_processors = [
    _nl_planning.response_processor,
    _code_execution.response_processor,
]
```

### 3.2 AutoFlow Extension

`auto_flow.py` adds one more processor at the beginning:

```python
request_processors = [
    agent_transfer.request_processor,   # 0. Agent transfer instructions + tool
    # ... then all SingleFlow processors
]
```

### 3.3 Processor #1: Basic Request (`basic.py:32–88`)

**Mutates**: `llm_request.model`, `llm_request.config`, `llm_request.live_connect_config`

```
1. model = agent.canonical_model
2. llm_request.model = model.model  (string name)
3. llm_request.config = deep copy of agent.generate_content_config (or new default)
4. IF agent.output_schema AND (no tools OR model supports output_schema+tools):
     llm_request.set_output_schema(agent.output_schema)
     → sets config.response_schema and config.response_mime_type = "application/json"
5. Copy all live_connect_config fields from run_config:
     response_modalities, speech_config, output_audio_transcription,
     input_audio_transcription, realtime_input_config, enable_affective_dialog,
     proactivity, session_resumption, context_window_compression
```

### 3.4 Processor #4: Instructions (`instructions.py:62–134`)

**Mutates**: `llm_request.config.system_instruction`, `llm_request.contents`

```
1. IF root_agent has global_instruction (deprecated):
     Resolve instruction, inject session state, append to system_instruction

2. IF agent has static_instruction:
     Convert to Content, append to system_instruction
     (non-text parts become user contents with references)

3. Instruction resolution depends on static_instruction presence:
   a. IF instruction AND NO static_instruction:
        Resolve instruction → inject state → append to system_instruction
   b. IF instruction AND static_instruction:
        Resolve instruction → inject state → append as user Content to llm_request.contents
```

**State injection** (`instructions_utils.inject_session_state`):
Replaces `{key}` placeholders in instruction text with values from `session.state`.

### 3.5 Processor #5: Identity (`identity.py:29–47`)

**Mutates**: `llm_request.config.system_instruction`

```python
si = f'You are an agent. Your internal name is "{agent.name}".'
if agent.description:
    si += f' The description about you is "{agent.description}".'
llm_request.append_instructions([si])
```

### 3.6 Processor #6: Contents (`contents.py:37–87`)

**Mutates**: `llm_request.contents` (complete replacement)

```
1. Save any existing contents (from instruction processor) as instruction_related_contents
2. IF agent.include_contents == 'default':
     llm_request.contents = _get_contents(branch, session.events, agent.name)
   ELSE ('none'):
     llm_request.contents = _get_current_turn_contents(branch, session.events, agent.name)
3. Re-insert instruction_related_contents at proper position
```

### 3.7 Processor #0: Agent Transfer (AutoFlow only, `agent_transfer.py:36–73`)

**Mutates**: `llm_request.config.system_instruction`, `llm_request.config.tools`, `llm_request.tools_dict`

```
1. Compute transfer_targets = _get_transfer_targets(agent)
2. IF no targets → return

3. Create TransferToAgentTool(agent_names=[...])
4. Append transfer instructions to system_instruction:
   - List of target agents with names and descriptions
   - Rules: transfer if another agent is better suited
   - NOTE listing valid agent names
   - If parent exists and transfers allowed: "transfer to parent if unsure"
5. Call transfer_tool.process_llm_request() → adds tool declaration
```

**Transfer target computation** (`agent_transfer.py:143–162`):

```
targets = []
targets.extend(agent.sub_agents)
IF parent exists AND is LlmAgent:
  IF NOT disallow_transfer_to_parent:
    targets.append(parent)
  IF NOT disallow_transfer_to_peers:
    targets.extend(parent.sub_agents WHERE name != self.name)
```

### 3.8 Processor #11: Output Schema Workaround (`_output_schema_processor.py:32–67`)

**Condition**: agent has output_schema AND tools AND model cannot use both together

```
1. Create SetModelResponseTool(agent.output_schema)
2. Append tool to llm_request
3. Append instruction: "IMPORTANT: provide final response using set_model_response tool"
```

### 3.9 Tool Registration in Preprocessing

After all processors run, `_preprocess_async()` (`base_llm_flow.py:550–581`):

```
FOR each tool_union in agent.tools:
  IF isinstance(BaseToolset):
    toolset.process_llm_request(tool_context, llm_request)

  tools = _convert_tool_union_to_tools(tool_union, context, model, multiple_tools)
  FOR each tool in tools:
    tool.process_llm_request(tool_context, llm_request)
```

Default `BaseTool.process_llm_request()` (`base_tool.py:115–129`):
```python
llm_request.append_tools([self])
```

`LlmRequest.append_tools()` (`llm_request.py:244–274`):
```
1. For each tool, get declaration = tool._get_declaration()
2. If declaration exists: add to tools_dict[tool.name] = tool
3. Find existing Tool with function_declarations in config.tools
   - If found: append declarations to it
   - If not found: create new Tool(function_declarations=[...])
```

---

## 4. Contents Construction Algorithm

### 4.1 Full History: `_get_contents()` (`contents.py:372–490`)

**Input**: `current_branch`, `session.events`, `agent_name`
**Output**: `list[types.Content]`

```
STEP 1: REWIND FILTERING
  Iterate events BACKWARD:
    IF event has rewind_before_invocation_id:
      Skip all events from that invocation_id forward to current position
    ELSE:
      Keep event
  Reverse to restore chronological order

STEP 2: BRANCH + VISIBILITY FILTERING
  Keep events WHERE _should_include_event_in_context(branch, event):
    NOT _contains_empty_content(event)
    AND NOT belongs to different branch
    AND NOT is auth event (adk_request_credential)
    AND NOT is confirmation event (adk_request_confirmation)
    AND NOT is framework event (adk_framework)
    AND NOT is input request event (adk_request_input)

STEP 3: COMPACTION HANDLING
  IF any event has actions.compaction:
    Process compaction events (replace ranges with summarized content)

STEP 4: TRANSCRIPTION AGGREGATION
  FOR each event:
    IF no content but has input_transcription:
      Accumulate text; when next event is not same type:
        Convert to Content(role='user', parts=[Part(text=accumulated)])
    IF no content but has output_transcription:
      Accumulate text; when next event is not same type:
        Convert to Content(role='model', parts=[Part(text=accumulated)])

STEP 5: OTHER-AGENT MESSAGE CONVERSION
  IF event is from another agent (author != agent_name AND author != 'user'):
    Convert to user-role content:
      - Text parts: "[agent_name] said: {text}"
      - Function calls: "[agent_name] called tool `name` with parameters: {args}"
      - Function responses: "[agent_name] `name` tool returned result: {response}"
      - Thoughts: EXCLUDED
      - If only "For context:" remains after filtering → skip event entirely

STEP 6: FUNCTION CALL/RESPONSE REARRANGEMENT
  A. _rearrange_events_for_latest_function_response():
     If last event is function_response but doesn't match second-to-last's calls:
       Find matching function_call event by scanning backward
       Collect all intermediate function_response events
       Merge them and place immediately after function_call event
       Remove all events between function_call and latest response

  B. _rearrange_events_for_async_function_responses_in_history():
     Build map: function_call_id → response_event_index
     Reorder so each function_call is immediately followed by its response
     If multiple response events for one call group → merge them

STEP 7: CONVERT TO CONTENTS
  FOR each event:
    content = deep_copy(event.content)
    IF NOT preserve_function_call_ids:
      Strip IDs starting with 'adk-' prefix (client-generated IDs)
    Append content to result list
```

### 4.2 Current Turn Only: `_get_current_turn_contents()` (`contents.py:493–533`)

```
Scan events BACKWARD to find latest event where:
  _should_include_event_in_context(branch, event)
  AND (author == 'user' OR is_other_agent_reply)

Call _get_contents() on events[found_index:] (slice from that point forward)
```

### 4.3 Event Filtering Predicates

**`_contains_empty_content()`** (`contents.py:262–285`):
```
Return True if:
  (no content OR no role OR no parts OR all parts are invisible)
  AND (no output_transcription AND no input_transcription)
  EXCEPT: events with compaction are never empty
```

**`_is_part_invisible()`** (`contents.py:233–259`):
```
Function calls and function responses → NEVER invisible
Otherwise invisible if:
  part.thought is True
  OR not (text or inline_data or file_data or executable_code or code_execution_result)
```

**`_is_event_belongs_to_branch()`** (`contents.py:677–692`):
```
IF no invocation_branch OR no event.branch → True (belongs)
Match: branch == event.branch OR branch.startswith(event.branch + '.')
```

### 4.4 Function Response Merging

`_merge_function_response_events()` (`contents.py:613–674`):

```
1. Start with deep copy of first response event
2. Build index: function_call_id → part position
3. For each subsequent event:
   For each part:
     IF function_response AND id already in index:
       REPLACE part at that index (update with latest response)
     ELIF function_response AND id NOT in index:
       APPEND part (new response)
     ELSE:
       APPEND part (non-response content)
```

---

## 5. LLM Call Mechanics

### 5.1 Call Chain

`_call_llm_async()` (`base_llm_flow.py:878–968`):

```
1. Run before_model_callback pipeline:
   a. plugin_manager.run_before_model_callback()
   b. IF no plugin override: iterate agent.canonical_before_model_callbacks
   c. IF any returns LlmResponse → yield it, skip LLM call, RETURN

2. Set labels: config.labels['adk_agent_name'] = agent.name

3. Get LLM instance: agent.canonical_model

4. Increment LLM call count (may raise LlmCallsLimitExceededError)

5. Call: llm.generate_content_async(llm_request, stream=streaming_mode==SSE)

6. Wrap with error handling: _run_and_handle_error()

7. For each llm_response:
   a. Run after_model_callback pipeline:
      - plugin_manager.run_after_model_callback()
      - IF no plugin override: iterate agent.canonical_after_model_callbacks
      - May add grounding_metadata from google_search_agent
   b. Yield llm_response
```

### 5.2 Streaming vs Non-Streaming

```
IF run_config.streaming_mode == StreamingMode.SSE:
  stream=True → yields partial LlmResponse objects
  Partial responses have partial=True
  Only non-partial or SSE-mode responses are yielded from _call_llm_async

IF streaming_mode != SSE:
  stream=False → single LlmResponse with complete content
```

### 5.3 LLM Response Creation

`LlmResponse.create()` (`llm_response.py:146–200`):

```
IF candidates exist:
  candidate = candidates[0]
  IF candidate has content+parts OR finish_reason==STOP:
    Return LlmResponse(content=candidate.content, grounding_metadata=..., ...)
  ELSE:
    Return LlmResponse(error_code=finish_reason, error_message=finish_message)
ELIF prompt_feedback exists:
  Return LlmResponse(error_code=block_reason, error_message=block_reason_message)
ELSE:
  Return LlmResponse(error_code='UNKNOWN_ERROR', error_message='Unknown error.')
```

### 5.4 Error Handling

`_run_and_handle_error()` (`base_llm_flow.py:1114–1190`):

```
TRY:
  Yield all responses from generator
EXCEPT Exception as model_error:
  Run on_model_error callbacks:
    1. plugin_manager.run_on_model_error_callback()
    2. agent.canonical_on_model_error_callbacks (iterate until non-None)
  IF callback returns LlmResponse → yield it
  ELSE → re-raise the error
```

### 5.5 LLM Call Count Enforcement

`InvocationContext.increment_llm_call_count()` (`invocation_context.py:306–317`):

Called before every non-CFC LLM call. Uses `_invocation_cost_manager` to check against `run_config.max_llm_calls`. Raises `LlmCallsLimitExceededError` if exceeded.

---

## 6. Postprocessing & Tool Dispatch

### 6.1 Postprocessing Pipeline

`_postprocess_async()` (`base_llm_flow.py:658–714`):

```
1. Run response_processors (NL planning, code execution)

2. IF no content AND no error_code AND not interrupted → RETURN (skip event)

3. Finalize model_response_event:
   Merge llm_response fields into event via model_validate()
   If function_calls present:
     - Populate client-generated IDs (adk-{uuid4})
     - Identify long_running_tool_ids

4. YIELD model_response_event

5. IF event has function_calls AND NOT partial:
   Run _postprocess_handle_function_calls_async()
```

### 6.2 Event Finalization

`_finalize_model_response_event()` (`base_llm_flow.py:80–113`):

```python
finalized_event = Event.model_validate({
    **model_response_event.model_dump(exclude_none=True),
    **llm_response.model_dump(exclude_none=True),
})
# Populate function call IDs and long-running tool info
if finalized_event.content:
    function_calls = finalized_event.get_function_calls()
    if function_calls:
        populate_client_function_call_id(finalized_event)  # adk-{uuid4} for null IDs
        finalized_event.long_running_tool_ids = get_long_running_function_calls(
            function_calls, llm_request.tools_dict
        )
```

### 6.3 Function Call Handling

`_postprocess_handle_function_calls_async()` (`base_llm_flow.py:825–867`):

```
1. Execute all function calls → get function_response_event
2. Generate auth event if any tool requested credentials
3. Generate confirmation event if any tool requested confirmation
4. YIELD auth_event (if any)
5. YIELD confirmation_event (if any)
6. YIELD function_response_event
7. IF set_model_response was called → yield final structured response event
8. IF transfer_to_agent in response actions:
   Find target agent, run its run_async(), yield all events
```

### 6.4 Tool Execution

`handle_function_calls_async()` (`functions.py:333–411`):

```
1. Get function_calls from event
2. Filter by provided filters (if any)
3. Create asyncio tasks for PARALLEL execution:
   tasks = [asyncio.create_task(_execute_single_function_call_async(...)) for fc in calls]
4. results = await asyncio.gather(*tasks)
5. Filter out None results
6. Merge parallel results: merge_parallel_function_response_events()
```

### 6.5 Single Tool Execution Lifecycle

`_execute_single_function_call_async()` (`functions.py:414–582`):

```
1. Deep copy function_call.args
2. Create ToolContext(invocation_context, function_call_id, tool_confirmation)
3. Look up tool in tools_dict:
   - If not found → run on_tool_error callbacks → raise ValueError

4. WITHIN tracing span 'execute_tool {name}':

   STEP 1: Plugin before_tool_callback
     response = plugin_manager.run_before_tool_callback(tool, args, tool_context)

   STEP 2: Agent before_tool_callbacks (if plugin didn't override)
     FOR callback in agent.canonical_before_tool_callbacks:
       response = callback(tool, args, tool_context)
       IF response → break

   STEP 3: Execute tool (if no callback override)
     TRY:
       response = await tool.run_async(args=args, tool_context=tool_context)
     EXCEPT:
       Run on_tool_error callbacks → use error response or re-raise

   STEP 4: Plugin after_tool_callback
     altered = plugin_manager.run_after_tool_callback(tool, args, tool_context, result)

   STEP 5: Agent after_tool_callbacks (if plugin didn't override)
     FOR callback in agent.canonical_after_tool_callbacks:
       altered = callback(tool, args, tool_context, tool_response=response)
       IF altered → break

   STEP 6: Use altered response if provided

   STEP 7: Long-running tool check
     IF tool.is_long_running AND no response → return None (no event)

   STEP 8: Build response event
```

### 6.6 Response Event Construction

`__build_response_event()` (`functions.py:939–967`):

```python
# Ensure result is a dict (spec requirement)
if not isinstance(function_result, dict):
    function_result = {'result': function_result}

part = Part.from_function_response(name=tool.name, response=function_result)
part.function_response.id = tool_context.function_call_id

content = Content(role='user', parts=[part])

event = Event(
    invocation_id=invocation_context.invocation_id,
    author=invocation_context.agent.name,
    content=content,
    actions=tool_context.actions,  # includes state_delta, transfer_to_agent, etc.
    branch=invocation_context.branch,
)
```

### 6.7 FunctionTool Execution

`FunctionTool.run_async()` (`function_tool.py:156–218`):

```
1. Preprocess args:
   - Convert JSON dicts to Pydantic models where type hints expect them
   - For Optional[PydanticModel]: unwrap Union to find model type

2. Filter args to only valid function parameters

3. Inject tool_context if function signature has 'tool_context' param

4. Check mandatory args:
   - Parameters without defaults (excluding *args, **kwargs)
   - If missing → return {'error': 'mandatory parameters not present...'}

5. Check require_confirmation:
   - If callable: evaluate with args
   - If True and no confirmation: request_confirmation(), return error
   - If True and confirmed=False: return {'error': 'rejected'}

6. Invoke function:
   - If async (coroutine or async __call__): await target(**args)
   - If sync: target(**args)
```

### 6.8 Parallel Function Response Merging

`merge_parallel_function_response_events()` (`functions.py:980–1020`):

```
IF single event → return it directly

Merge parts from all events into one list
Merge actions via deep_merge_dicts (recursive dict merge)
Create merged Event with combined parts and actions
Preserve timestamp from first event
```

### 6.9 Client Function Call ID Management

- **Generate**: `f'adk-{uuid.uuid4()}'` (`functions.py:164–165`)
- **Populate**: Assign to function_calls with `id=None` (`functions.py:168–173`)
- **Strip**: Before sending to LLM, remove IDs starting with `'adk-'` from both function_call.id and function_response.id (`functions.py:176–198`)

---

## 7. Event System & Session Persistence

### 7.1 Event Schema

`Event` extends `LlmResponse` (`event.py:30–128`):

```
Fields from LlmResponse:
  content: Optional[Content]
  grounding_metadata: Optional[GroundingMetadata]
  partial: Optional[bool]
  turn_complete: Optional[bool]
  finish_reason: Optional[FinishReason]
  error_code: Optional[str]
  error_message: Optional[str]
  interrupted: Optional[bool]
  custom_metadata: Optional[dict[str, Any]]
  usage_metadata: Optional[GenerateContentResponseUsageMetadata]
  input_transcription: Optional[Transcription]
  output_transcription: Optional[Transcription]
  avg_logprobs: Optional[float]
  logprobs_result: Optional[LogprobsResult]
  cache_metadata: Optional[CacheMetadata]
  citation_metadata: Optional[CitationMetadata]
  interaction_id: Optional[str]
  model_version: Optional[str]

Event-specific fields:
  invocation_id: str = ''
  author: str                          # 'user' or agent name
  actions: EventActions                # default EventActions()
  long_running_tool_ids: Optional[set[str]] = None
  branch: Optional[str] = None        # e.g. "agent_1.agent_2.agent_3"
  id: str = ''                         # auto-generated uuid4
  timestamp: float                     # datetime.now().timestamp()
```

### 7.2 EventActions Schema

`EventActions` (`event_actions.py:50–110`):

```
skip_summarization: Optional[bool] = None     # Skip model summarization of tool response
state_delta: dict[str, object] = {}           # Session state changes
artifact_delta: dict[str, int] = {}           # Artifact version changes
transfer_to_agent: Optional[str] = None       # Target agent name
escalate: Optional[bool] = None               # Escalate to parent
requested_auth_configs: dict[str, AuthConfig] = {}     # Auth requests by function_call_id
requested_tool_confirmations: dict[str, ToolConfirmation] = {}  # Confirmation requests
compaction: Optional[EventCompaction] = None   # Event compaction data
end_of_agent: Optional[bool] = None            # Agent finished its run
agent_state: Optional[dict[str, Any]] = None   # Agent state checkpoint
rewind_before_invocation_id: Optional[str] = None  # Rewind target
```

### 7.3 Persistence Rules

Events are appended via `session_service.append_event()`. Rules (`runners.py:702–831`):

| Condition | Persisted? |
|---|---|
| `event.partial is True` | **No** |
| Live audio with inline_data | **No** |
| Live audio with file_data (artifact ref) | Yes |
| Function call/response | Yes |
| Transcription (non-partial) | Yes |
| Normal model response | Yes |
| User message | Yes (before agent runs) |

### 7.4 State Delta Application

State deltas in `event.actions.state_delta` are key-value pairs that modify `session.state`. They are applied by the session service when processing the event (not in the flow itself). Setting a value to `None` removes the key.

---

## 8. Agent Transfer Mechanism

### 8.1 TransferToAgentTool

`transfer_to_agent_tool.py`:

```python
def transfer_to_agent(agent_name: str, tool_context: ToolContext):
    """Transfer to agent with given name."""
    tool_context.actions.transfer_to_agent = agent_name

class TransferToAgentTool(FunctionTool):
    def __init__(self, agent_names: list[str]):
        super().__init__(transfer_to_agent)
        self._agent_names = agent_names

    def _get_declaration(self):
        decl = super()._get_declaration()
        # Add enum constraint to agent_name parameter
        decl.parameters.properties['agent_name'].enum = self._agent_names
        return decl
```

### 8.2 Transfer Execution Flow

After tool dispatch, in `_postprocess_handle_function_calls_async()` (`base_llm_flow.py:860–867`):

```python
transfer_to_agent = function_response_event.actions.transfer_to_agent
if transfer_to_agent:
    agent_to_run = self._get_agent_to_run(invocation_context, transfer_to_agent)
    # Run the target agent and yield all its events
    async for event in agent_to_run.run_async(invocation_context):
        yield event
```

`_get_agent_to_run()` (`base_llm_flow.py:869–876`):
```python
root_agent = invocation_context.agent.root_agent
agent_to_run = root_agent.find_agent(agent_name)
if not agent_to_run:
    raise ValueError(f'Agent {agent_name} not found in the agent tree.')
```

### 8.3 Transfer Target Computation

`_get_transfer_targets()` (`agent_transfer.py:143–162`):

```
result = list(agent.sub_agents)

IF parent_agent exists AND is LlmAgent:
  IF NOT disallow_transfer_to_parent:
    result.append(parent_agent)
  IF NOT disallow_transfer_to_peers:
    result.extend(
      peer for peer in parent.sub_agents
      if peer.name != agent.name
    )
```

### 8.4 Transfer Instructions Injected

`_build_transfer_instructions()` (`agent_transfer.py:86–140`):

```
"You have a list of other agents to transfer to:

Agent name: {name}
Agent description: {description}
[... for each target ...]

If you are the best to answer the question according to your description,
you can answer it.

If another agent is better for answering the question according to its
description, call `transfer_to_agent` function to transfer the question to that
agent. When transferring, do not generate any text other than the function call.

**NOTE**: the only available agents for `transfer_to_agent` function are
`agent_a`, `agent_b`, `agent_c`.

[If parent exists and transfers allowed:]
If neither you nor the other agents are best for the question, transfer to your
parent agent {parent_name}."
```

---

## 9. Callback System — Complete Reference

### 9.1 All Callback Hook Points

| Hook | Location | Signature | Non-None Return Effect |
|---|---|---|---|
| `before_agent_callback` | `base_agent.py:270` | `(callback_context) → Optional[Content]` | Skip agent run, yield content as event |
| `after_agent_callback` | `base_agent.py:300` | `(callback_context) → Optional[Content]` | Yield additional event after agent |
| `before_model_callback` | `base_llm_flow.py:970` | `(callback_context, llm_request) → Optional[LlmResponse]` | Skip LLM call, use returned response |
| `after_model_callback` | `base_llm_flow.py:1007` | `(callback_context, llm_response) → Optional[LlmResponse]` | Replace LLM response |
| `on_model_error_callback` | `base_llm_flow.py:1141` | `(callback_context, llm_request, error) → Optional[LlmResponse]` | Use returned response instead of raising |
| `before_tool_callback` | `functions.py:498` | `(tool, args, tool_context) → Optional[dict]` | Skip tool execution, use as result |
| `after_tool_callback` | `functions.py:539` | `(tool, args, tool_context, tool_response) → Optional[dict]` | Replace tool result |
| `on_tool_error_callback` | `functions.py:423` | `(tool, args, tool_context, error) → Optional[dict]` | Use returned dict instead of raising |

### 9.2 Callback Execution Order

For all callbacks, the order is:
```
1. Plugin callbacks (plugin_manager.run_*_callback)
   → If returns non-None, use it and skip agent callbacks
2. Agent canonical callbacks (list, iterated in order)
   → First non-None return wins
```

Both sync and async callbacks are supported — `inspect.isawaitable()` is used to detect and await async results.

---

## 10. Multi-Agent Orchestration

### 10.1 Branch-Based Context Isolation

Each agent gets a `branch` value when `_create_invocation_context()` is called:

```python
# In BaseAgent._create_invocation_context():
if parent_context.branch:
    ctx.branch = f"{parent_context.branch}.{self.name}"
else:
    ctx.branch = self.name
```

Branch matching (`contents.py:677–692`):
- No branch on either side → visible
- Exact match → visible
- Current branch starts with `event.branch + '.'` → visible (parent events visible to children)
- Otherwise → invisible

### 10.2 Resumable Invocations

When `resumability_config.is_resumable` is True:

**Pause conditions** (`invocation_context.py:355–389`):
```
should_pause IF:
  is_resumable
  AND event.long_running_tool_ids is non-empty
  AND event has function_calls
  AND at least one function_call.id is in long_running_tool_ids
```

**Resume flow** (`llm_agent.py:448–485`):
```
1. Load agent_state
2. IF state exists AND sub-agent needs resuming:
   - Determine sub-agent via _get_subagent_to_resume()
   - Run it, yield events, mark end_of_agent
3. Resume check in _run_one_step_async:
   - If last event has function_calls and is_resumable:
     Execute those calls directly (skip LLM)
```

**Sub-agent resume logic** (`llm_agent.py:705–748`):
```
Get events for current invocation + branch
IF last event is from this agent:
  Return transfer_to_agent target (if any) or None
IF last event is from user:
  Find matching function_call
  If it's from this agent → no sub-agent to resume (this agent handles it)
For other cases: scan backward for last transfer_to_agent from this agent
```

---

## 11. Data Flow Summary — One Complete ReAct Cycle

```
User Message
    │
    ▼
Runner.run_async()
    │ create InvocationContext
    │ append user Event to session
    │ select agent to run
    ▼
BaseAgent.run_async()
    │ before_agent_callback
    ▼
LlmAgent._run_async_impl()
    │
    ▼
BaseLlmFlow.run_async()          ◄── OUTER LOOP
    │
    ▼
_run_one_step_async()             ◄── ONE LLM CALL
    │
    ├─► PREPROCESS
    │   ├── basic: set model, config
    │   ├── auth: resolve credentials
    │   ├── confirmation: handle pending confirmations
    │   ├── instructions: build system_instruction
    │   ├── identity: add agent identity
    │   ├── contents: build conversation history
    │   ├── [agent_transfer: if AutoFlow]
    │   ├── context_cache, interactions, nl_planning
    │   ├── code_execution
    │   └── output_schema: workaround if needed
    │   └── tool registration: tool.process_llm_request()
    │
    ├─► LLM CALL
    │   ├── before_model_callback (may skip call)
    │   ├── llm.generate_content_async()
    │   ├── after_model_callback (may alter response)
    │   └── on_model_error (fallback on failure)
    │
    ├─► POSTPROCESS
    │   ├── response_processors (NL planning, code execution)
    │   ├── Finalize event (merge response into Event)
    │   ├── Populate function call IDs
    │   ├── YIELD model_response_event
    │   │
    │   └── IF function_calls present:
    │       ├── Execute tools IN PARALLEL (asyncio.gather)
    │       │   ├── before_tool_callback
    │       │   ├── tool.run_async()
    │       │   ├── after_tool_callback
    │       │   └── build function_response Event
    │       ├── Merge parallel responses
    │       ├── YIELD auth_event (if any)
    │       ├── YIELD confirmation_event (if any)
    │       ├── YIELD function_response_event
    │       └── IF transfer_to_agent:
    │           └── Run target agent, yield its events
    │
    ▼
CHECK is_final_response()
    ├── True  → EXIT loop
    └── False → CONTINUE loop (go back to _run_one_step_async)

    ▼
after_agent_callback
    │
    ▼
Runner persists non-partial events
    │
    ▼
Events yielded to caller
```

---

## Appendix A: Complete Request Processor Chain

### SingleFlow Order

| # | Processor | File | Mutates |
|---|---|---|---|
| 1 | `basic.request_processor` | `basic.py` | model, config, live_connect_config |
| 2 | `auth_preprocessor.request_processor` | `auth/auth_preprocessor.py` | credentials resolution |
| 3 | `request_confirmation.request_processor` | `request_confirmation.py` | pending confirmations |
| 4 | `instructions.request_processor` | `instructions.py` | system_instruction, contents |
| 5 | `identity.request_processor` | `identity.py` | system_instruction |
| 6 | `contents.request_processor` | `contents.py` | contents (full replacement) |
| 7 | `context_cache_processor.request_processor` | `context_cache_processor.py` | cache config |
| 8 | `interactions_processor.request_processor` | `interactions_processor.py` | previous_interaction_id |
| 9 | `_nl_planning.request_processor` | `_nl_planning.py` | thought markers |
| 10 | `_code_execution.request_processor` | `_code_execution.py` | contents optimization |
| 11 | `_output_schema_processor.request_processor` | `_output_schema_processor.py` | tools, system_instruction |

### AutoFlow Order (prepends one processor)

| # | Processor | File | Mutates |
|---|---|---|---|
| 0 | `agent_transfer.request_processor` | `agent_transfer.py` | system_instruction, tools |
| 1–11 | (same as SingleFlow) | | |

### Response Processor Order

| # | Processor | File |
|---|---|---|
| 1 | `_nl_planning.response_processor` | `_nl_planning.py` |
| 2 | `_code_execution.response_processor` | `_code_execution.py` |

---

## Appendix B: Event Schema Reference

```
Event (extends LlmResponse):
  ┌─ Identity ─────────────────────────────┐
  │ id: str                    [uuid4]      │
  │ invocation_id: str         [e-uuid4]    │
  │ author: str                [user|agent] │
  │ timestamp: float           [epoch]      │
  │ branch: Optional[str]     [a.b.c]      │
  ├─ Content ──────────────────────────────┤
  │ content: Optional[Content]              │
  │   .role: str               [user|model] │
  │   .parts: list[Part]                    │
  │     .text: Optional[str]                │
  │     .function_call: Optional[FC]        │
  │       .name: str                        │
  │       .args: dict                       │
  │       .id: Optional[str]  [adk-uuid4]   │
  │     .function_response: Optional[FR]    │
  │       .name: str                        │
  │       .response: dict                   │
  │       .id: Optional[str]  [adk-uuid4]   │
  │     .inline_data: Optional[Blob]        │
  │     .file_data: Optional[FileData]      │
  │     .executable_code: Optional[...]     │
  │     .code_execution_result: Optional[.] │
  │     .thought: Optional[bool]            │
  ├─ Actions ──────────────────────────────┤
  │ actions: EventActions                   │
  │   .skip_summarization: Optional[bool]   │
  │   .state_delta: dict[str, object]       │
  │   .artifact_delta: dict[str, int]       │
  │   .transfer_to_agent: Optional[str]     │
  │   .escalate: Optional[bool]             │
  │   .requested_auth_configs: dict         │
  │   .requested_tool_confirmations: dict   │
  │   .compaction: Optional[EventCompaction]│
  │   .end_of_agent: Optional[bool]         │
  │   .agent_state: Optional[dict]          │
  │   .rewind_before_invocation_id: Opt[str]│
  ├─ Metadata ─────────────────────────────┤
  │ partial: Optional[bool]                 │
  │ turn_complete: Optional[bool]           │
  │ finish_reason: Optional[FinishReason]   │
  │ error_code: Optional[str]               │
  │ error_message: Optional[str]            │
  │ interrupted: Optional[bool]             │
  │ custom_metadata: Optional[dict]         │
  │ usage_metadata: Optional[UsageMetadata] │
  │ grounding_metadata: Optional[...]       │
  │ input_transcription: Optional[Transc]   │
  │ output_transcription: Optional[Transc]  │
  │ long_running_tool_ids: Optional[set]    │
  │ cache_metadata: Optional[CacheMetadata] │
  │ citation_metadata: Optional[...]        │
  │ interaction_id: Optional[str]           │
  │ model_version: Optional[str]            │
  │ avg_logprobs: Optional[float]           │
  │ logprobs_result: Optional[...]          │
  └────────────────────────────────────────┘
```

---

## Appendix C: LlmRequest Schema Reference

```
LlmRequest:
  model: Optional[str]                    # Model name string
  contents: list[Content]                 # Conversation history
  config: GenerateContentConfig           # Full generation config
    .system_instruction: Optional[str]    # System prompt (concatenated with \n\n)
    .tools: Optional[list[Tool]]          # Tool declarations
    .response_schema: Optional[type]      # Output schema (Pydantic model)
    .response_mime_type: Optional[str]    # "application/json" when schema set
    .labels: Optional[dict[str, str]]     # Includes adk_agent_name
  live_connect_config: LiveConnectConfig  # Live/bidi streaming config
  tools_dict: dict[str, BaseTool]         # Runtime tool lookup (excluded from serialization)
  cache_config: Optional[ContextCacheConfig]
  cache_metadata: Optional[CacheMetadata]
  previous_interaction_id: Optional[str]
```

**`append_instructions(list[str])`**: Concatenates strings with `\n\n`, appends to `config.system_instruction`.

**`append_instructions(Content)`**: Extracts text parts to system_instruction, creates user contents with references for non-text parts (inline_data, file_data).

**`append_tools(list[BaseTool])`**: Gets declarations, adds to tools_dict, appends to existing Tool with function_declarations or creates new one.

**`set_output_schema(BaseModel)`**: Sets `config.response_schema = model`, `config.response_mime_type = "application/json"`.
