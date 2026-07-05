# Agentic Graph RAG - AI Agent Instructions

## System Architecture

**Core Pattern**: Multi-agent orchestrator with lazy-loaded LLM backends and dynamic workflow execution.

### Main Flow: `orchestrator.py` → `state.py`
- **Orchestrator**: State machine with 8 steps (START → UNDERSTAND → RETRIEVE → ASSESS → REFRESH → REASON → VALIDATE → ANSWER)
- **AgentState**: Single dataclass carrying all context through pipeline (query, retrieval results, reasoning trace, confidence scores)
- **FlowStep Enum**: Defines valid state transitions (see `_transitions` dict)
- **Lazy Loading**: Models load only via factories (e.g., `config["phi4_mini_factory"]()` as context manager) and unload immediately after use

### Data Flow Pipeline
```
Query → Phi4-mini (understand + plan) 
  → RetrieverAgent (vector/graph hybrid search)
  → ValidatorAgent (assess confidence)
  → KnowledgeRefreshAgent (fetch external data if gaps detected)
  → ReasonerAgent (generate answer with citations)
  → ValidatorAgent (final scoring)
```

### Key Agents (`agents/`)
- **Orchestrator**: State machine orchestrator with context managers for model lifecycle
- **RetrieverAgent**: Hybrid search (vector via Weaviate + graph via KuzuDB with k-hop reasoning)
- **ReasonerAgent**: LLM-based answer generation with citation tracking
- **ValidatorAgent**: Confidence scoring and quality assessment
- **KnowledgeRefreshAgent** / **ImprovedKnowledgeRefreshAgent**: External data ingestion (web search, API calls)
- **AdaptiveOrchestrator**: Advanced routing with dynamic step skipping and early exit (see `DynamicPlan`)
- **QueryClassifier**: Query type/complexity analysis for routing decisions

## Critical Developer Patterns

### 1. **Lazy Model Loading - ESSENTIAL**
Every heavyweight model loads inside an async context manager:
```python
async with self.config["phi4_mini_factory"]() as phi4:
    result = await phi4.understand_and_plan(query, history)
# Model automatically unloads here (memory freed)
```
**Why**: Apple Silicon (MLX backend) + CPU memory constraints. Loading multiple models simultaneously causes OOM.

### 2. **AgentState is Immutable During Step**
Pass state through method calls; store results in `state.` fields:
```python
state.retrieval = await self.retriever.retrieve(state, query)
state.final_answer = result["answer"]
state.reasoning_trace.append(step_reason)
```
Don't create temporary objects outside state; use `AgentState` as central data holder.

### 3. **Config-Driven Factories (No Direct Instantiation)**
In `__init__`, never instantiate heavyweight components directly. Instead, use factories:
```python
# ✅ Correct
self.config["llm_client_factory"]  # Just store function reference

# ❌ Wrong
self.llm_client = Granite7BClient()  # Models load immediately → OOM
```
**Location**: Factories defined in `configs/main_config.py`

### 4. **Tool Registry Pattern**
Register tools as methods with `@register_tool()` decorator:
```python
@register_tool("retrieve")
async def tool_retrieve(self, state: AgentState, query: str = "", **_):
    state.retrieval = await self.retriever.retrieve(state, query or state.query)
```
Used by `Phi4MiniClient.understand_and_plan()` for tool selection and by `SkillExecutor` for execution.

### 5. **Transition Validation**
Check `self._transitions` dict before changing state:
```python
allowed = self._transitions.get(state.current_step, [])
if next_step not in allowed:
    state.current_step = FlowStep.ERROR
```

## Integration Points

### Vector Store: Weaviate
- **Purpose**: Semantic search with MMR reranking
- **Location**: `vector_store/weaviate_client.py` + `RetrieverAgent._vector_search()`
- **Config**: `config["weaviate"]["mode"]` ("embedded" for local testing, "docker" for prod)
- **Usage**: Returns `{"chunks": [...], "scores": [...]}`

### Knowledge Graph: KuzuDB
- **Purpose**: Multi-hop entity reasoning + relationship traversal
- **Location**: `knowledg_graph/kuzudb_package/async_manager.py`
- **Graph Hops**: `config["graph_hops"]` (default 2)
- **Usage**: `await manager.get_subgraph(entity_ids, depth=2)` returns nodes + edges

### LLM Backends (configured via `llm/prompt_understanding.py`)
- **Phi4-mini** (MLX): Query understanding, plan generation, gap detection
- **Granite-7b** (MLX): Reasoning and answer generation
- **Backend abstraction**: `OllamaBackend` | `LlamaCppBackend` | `MLXBackend`
- **Config selection**: `config["phi4_mini"]["backend"]` = "mlx" | "ollama" | "llama_cpp"

### Query Analysis: QueryClassifier
- **Input**: Query string
- **Output**: `QueryAnalysis(type, complexity, keywords, temporal_indicators)`
- **Location**: `agents/query_classifier.py`
- **Used by**: `AdaptiveOrchestrator` for routing decisions

## Workflow & Debugging

### Run Main UI
```bash
python main.py  # Gradio interface on http://localhost:7860
```

### Run Adaptive Mode
```bash
python main_adaptive_enhanced.py
```

### Key Config File
- `configs/main_config.py`: All paths, thresholds, model configs
- Modify `"confidence_threshold": 0.65` to control answer quality bar
- Add models via `def make_<model>_client()` and register in dict

### Memory Monitoring
- `core/memory_monitor.py`: Tracks GPU/CPU usage
- `core/model_manager.py`: Lazy load/unload orchestration
- Enable verbose logging: `logger.setLevel(logging.DEBUG)`

### Common Debugging
1. **Agent stuck in loop**: Check `max_iterations` (default 2) in `AgentState`
2. **Low answer quality**: Lower `confidence_threshold` in config or add REFRESH step
3. **OOM errors**: Verify context managers close (use `async with ...`) and check `model_manager.py` for leaks
4. **Slow retrieval**: Check `config["vector_top_k"]` (default 10) and `config["graph_hops"]` (default 2)

## Advanced Patterns

### Dynamic Routing (AdaptiveOrchestrator)
- **DynamicPlan**: Skip unnecessary steps based on query type
  - `skip_steps`: [FlowStep.ASSESS] for simple factual queries
  - `allow_early_exit`: true if high confidence before reasoning
- **Strategy**: "sequential" (default) | "parallel" (future)

### Custom Skill Integration (SkillExecutor)
Register real skill implementations in `AdaptiveOrchestrator._register_real_skills()`:
```python
async def real_search(query: str, sources: list = None) -> dict:
    # Hybrid vector + graph + web search
    results = await orch.retriever.retrieve(state, query)
    return {"results": results.vector_chunks}

registry["search"] = real_search
```

### Multi-Language Support
- Phi4-mini detects language in `result["language"]`
- Store in `state.language` for downstream processing
- Note: Some components assume English (NER, entity linking)

## File Reference Map

- **Core orchestration**: `agents/orchestrator.py`, `agents/adaptive_orchestrator.py`
- **State & flow**: `agents/state.py`, `agents/query_classifier.py`
- **Retrieval**: `agents/retriever_agent.py` (hybrid search logic)
- **LLM backends**: `llm/prompt_understanding.py` (Phi4) + `llm/granite_client/mlx_client.py` (Granite)
- **Graph DB**: `knowledg_graph/kuzudb_package/async_manager.py`
- **Vector store**: `vector_store/weaviate_client.py`
- **Config**: `configs/main_config.py`
- **Memory**: `core/model_manager.py`, `core/memory_monitor.py`
- **UI**: `api/dashboard.py`, `main.py` (Gradio), `main_adaptive_enhanced.py`

## Documentation
- **Architecture**: `/docs/COMPLETE_FLOW_ANALYSIS.md`
- **Adaptive mode**: `/docs/ADAPTIVE_MAIN_GUIDE.md`
- **Graph setup**: `/docs/KUZU_SETUP_GUIDE.md`
- **Skills system**: `/docs/SKILL_TOOL_CALL_SYSTEM.md`
