# سیستم Skill/Tool Call در AdaptiveOrchestrator

## خلاصه اجرایی

این سند سیستم جامع skill/tool call را که به AdaptiveOrchestrator و phi4.understand_and_plan اضافه شده، شرح می‌دهد.

**نتیجه**: ✅ سیستم کامل و یکپارچه با قابلیت اجرای tool calls در plan steps

---

## 1. معماری کلی

### 1.1 اجزای سیستم

```
┌─────────────────────────────────────────────────────────────┐
│                    AdaptiveOrchestrator                      │
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │ Query        │      │ phi4.        │                     │
│  │ Classifier   │─────▶│ understand_  │                     │
│  │              │      │ and_plan     │                     │
│  └──────────────┘      └──────┬───────┘                     │
│                               │                              │
│                               ▼                              │
│                    ┌──────────────────┐                     │
│                    │  tool_calls      │                     │
│                    │  plan_steps      │                     │
│                    └────────┬─────────┘                     │
│                             │                               │
│                             ▼                               │
│                  ┌─────────────────────┐                   │
│                  │  SkillExecutor      │                   │
│                  │  - execute_skill    │                   │
│                  │  - execute_steps    │                   │
│                  └─────────┬───────────┘                   │
│                            │                                │
│                            ▼                                │
│                  ┌─────────────────────┐                   │
│                  │  SkillRegistry      │                   │
│                  │  - search           │                   │
│                  │  - web_search       │                   │
│                  │  - graph_query      │                   │
│                  │  - calculate        │                   │
│                  └─────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 فلوی اجرا

```
1. Query → QueryClassifier
   ↓
2. phi4.understand_and_plan
   ├─ tool_calls: [{"tool": "web_search", "args": {...}}]
   └─ plan_steps: [{"id": 1, "action": "retrieve", "tool": null}]
   ↓
3. Execute tool_calls (if any)
   ↓
4. Execute plan_steps (if any)
   ↓
5. Generate Dynamic Plan
   ↓
6. Execute Dynamic Plan
```

---

## 2. SkillRegistry

### 2.1 Skills پیش‌فرض

#### search
```python
async def search_skill(query: str, sources: List[str] = None) -> Dict[str, Any]:
    """جستجو در منابع مختلف (vector, graph, bm25)"""
    return {
        "query": query,
        "sources": sources or ["vector", "graph"],
        "results": []
    }
```

#### web_search
```python
async def web_search_skill(query: str, num_results: int = 5) -> Dict[str, Any]:
    """جستجوی وب"""
    return {
        "query": query,
        "num_results": num_results,
        "results": []
    }
```

#### graph_query
```python
async def graph_query_skill(cypher: str) -> Dict[str, Any]:
    """کوئری از knowledge graph"""
    return {
        "cypher": cypher,
        "results": []
    }
```

#### calculate
```python
async def calculate_skill(expression: str) -> Dict[str, Any]:
    """محاسبات ریاضی"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}
}
```

### 2.2 ثبت Skill جدید

```python
from agents.skill_executor import register_skill, SkillType

@register_skill(
    "my_custom_skill",
    "Does something useful",
    {"param1": "string", "param2": "int"},
    SkillType.CUSTOM
)
async def my_custom_skill(param1: str, param2: int):
    # پیاده‌سازی
    return {"result": f"{param1} - {param2}"}
```

---

## 3. SkillExecutor

### 3.1 اجرای تک Skill

```python
executor = SkillExecutor()

result = await executor.execute_skill(
    skill_name="web_search",
    args={"query": "latest AI news", "num_results": 5}
)

if result.success:
    print(f"Output: {result.output}")
    print(f"Time: {result.execution_time:.2f}s")
else:
    print(f"Error: {result.error}")
```

### 3.2 اجرای چند Skill

```python
skill_calls = [
    {"tool": "search", "args": {"query": "quantum computing"}},
    {"tool": "web_search", "args": {"query": "quantum news"}},
]

# ترتیبی
results = await executor.execute_multiple_skills(skill_calls, parallel=False)

# موازی
results = await executor.execute_multiple_skills(skill_calls, parallel=True)
```

### 3.3 اجرای Plan Steps

```python
step = {
    "id": 1,
    "action": "retrieve",
    "tool": "search",
    "args": {"query": "AI"}
}

result = await executor.execute_step_with_skill(step)
```

---

## 4. یکپارچگی با phi4.understand_and_plan

### 4.1 خروجی phi4

```python
result = await phi4.understand_and_plan(query, history)

# tool_calls
result["tool_calls"] = [
    {
        "tool": "web_search",
        "args": {"query": "latest AI news"},
        "reason": "Need real-time information"
    }
]

# plan_steps
result["steps"] = [
    {
        "id": 1,
        "action": "retrieve",
        "description": "Retrieve information",
        "depends_on": [],
        "tool": None,  # یا نام tool
        "args": {}
    },
    {
        "id": 2,
        "action": "reason",
        "description": "Analyze information",
        "depends_on": [1],
        "tool": None,
        "args": {}
    }
]
```

### 4.2 اجرا در Orchestrator

```python
# در _step_understand_and_analyze

# 1. اجرای tool_calls
if state.tool_calls:
    await self._execute_tool_calls(state)
    # نتایج در state.tool_results

# 2. اجرای plan_steps
if state.plan_steps:
    await self._execute_plan_steps(state)
    # نتایج در state.step_results
```

---

## 5. AgentState بروزرسانی شده

### 5.1 فیلدهای جدید

```python
@dataclass
class AgentState:
    # ... فیلدهای قبلی
    
    # Tool/Skill Execution
    tool_results: list[dict] = field(default_factory=list)
    step_results: dict[int, dict] = field(default_factory=dict)
    
    # Validation
    validation: Any = None
```

### 5.2 tool_results

```python
state.tool_results = [
    {
        "tool": "web_search",
        "success": True,
        "output": {"results": [...]},
        "error": None,
        "execution_time": 1.23
    }
]
```

### 5.3 step_results

```python
state.step_results = {
    1: {
        "success": True,
        "output": {"chunks": 10, "confidence": 0.85},
        "error": None
    },
    2: {
        "success": True,
        "output": {"reasoning": "..."},
        "error": None
    }
}
```

---

## 6. Actions پشتیبانی شده

### 6.1 retrieve
```python
{
    "id": 1,
    "action": "retrieve",
    "description": "Retrieve information",
    "tool": None,  # از retriever agent استفاده می‌کند
    "args": {}
}
```

### 6.2 search
```python
{
    "id": 2,
    "action": "search",
    "description": "Search for information",
    "tool": "search",  # یا None برای استفاده از skill پیش‌فرض
    "args": {"query": "AI", "sources": ["vector", "graph"]}
}
```

### 6.3 reason
```python
{
    "id": 3,
    "action": "reason",
    "description": "Analyze and synthesize",
    "tool": None,  # از reasoner agent استفاده می‌کند
    "args": {},
    "depends_on": [1, 2]
}
```

### 6.4 validate
```python
{
    "id": 4,
    "action": "validate",
    "description": "Validate reasoning",
    "tool": None,  # از validator agent استفاده می‌کند
    "args": {},
    "depends_on": [3]
}
```

### 6.5 summarize
```python
{
    "id": 5,
    "action": "summarize",
    "description": "Summarize information",
    "tool": None,
    "args": {}
}
```

### 6.6 compare
```python
{
    "id": 6,
    "action": "compare",
    "description": "Compare two concepts",
    "tool": None,
    "args": {}
}
```

---

## 7. مثال‌های کاربردی

### 7.1 Query با Web Search

```python
query = "What is the latest news about AI?"

# phi4 output:
{
    "tool_calls": [
        {
            "tool": "web_search",
            "args": {"query": "latest AI news", "num_results": 5},
            "reason": "Need real-time information"
        }
    ],
    "steps": [
        {"id": 1, "action": "retrieve", "tool": None},
        {"id": 2, "action": "reason", "tool": None, "depends_on": [1]}
    ]
}

# Execution:
# 1. Execute web_search tool
# 2. Execute retrieve action
# 3. Execute reason action
# 4. Generate answer
```

### 7.2 Query با محاسبات

```python
query = "Calculate 15% of 250"

# phi4 output:
{
    "tool_calls": [
        {
            "tool": "calculate",
            "args": {"expression": "250 * 0.15"},
            "reason": "Mathematical calculation needed"
        }
    ],
    "steps": [
        {"id": 1, "action": "reason", "tool": None}
    ]
}

# Execution:
# 1. Execute calculate tool → result: 37.5
# 2. Execute reason action
# 3. Generate answer with calculation result
```

### 7.3 Query پیچیده با چند Step

```python
query = "Compare quantum computing and classical computing"

# phi4 output:
{
    "tool_calls": [],
    "steps": [
        {
            "id": 1,
            "action": "retrieve",
            "description": "Retrieve info about quantum computing",
            "tool": "search",
            "args": {"query": "quantum computing"},
            "depends_on": []
        },
        {
            "id": 2,
            "action": "retrieve",
            "description": "Retrieve info about classical computing",
            "tool": "search",
            "args": {"query": "classical computing"},
            "depends_on": []
        },
        {
            "id": 3,
            "action": "reason",
            "description": "Analyze both concepts",
            "tool": None,
            "depends_on": [1, 2]
        },
        {
            "id": 4,
            "action": "compare",
            "description": "Compare the two",
            "tool": None,
            "depends_on": [3]
        }
    ]
}

# Execution:
# 1. Execute step 1 (retrieve quantum)
# 2. Execute step 2 (retrieve classical)
# 3. Execute step 3 (reason) - waits for 1,2
# 4. Execute step 4 (compare) - waits for 3
# 5. Generate answer
```

---

## 8. Error Handling

### 8.1 Tool Execution Failure

```python
result = await executor.execute_skill("web_search", {"query": "test"})

if not result.success:
    logger.error(f"Tool failed: {result.error}")
    # Fallback یا retry
```

### 8.2 Step Dependency Failure

```python
# اگر step 1 fail شود، step 2 که به آن وابسته است اجرا نمی‌شود
step_results = {
    1: {"success": False, "error": "Network error"},
    2: {"success": False, "error": "Missing dependencies: [1]"}
}
```

### 8.3 Unknown Tool

```python
result = await executor.execute_skill("unknown_tool", {})
# result.success = False
# result.error = "Skill 'unknown_tool' not found"
```

---

## 9. Monitoring و Logging

### 9.1 Execution Summary

```python
summary = executor.get_execution_summary()

print(f"Total executions: {summary['total_executions']}")
print(f"Success rate: {summary['success_rate']:.2%}")
print(f"Average time: {summary['average_time']:.2f}s")
```

### 9.2 Logs

```
[INFO] Executing skill: web_search with args: {'query': 'AI news'}
[INFO] Skill executed successfully: web_search in 1.23s
[INFO] Executing 3 plan steps
[INFO] Executing step 1: retrieve
[INFO] Step 1 completed successfully
[INFO] Executing step 2: reason
[INFO] Step 2 completed successfully
```

---

## 10. Best Practices

### 10.1 ثبت Skills

```python
# ✅ Good: با type hints و docstring
@register_skill("my_skill", "Clear description", {...}, SkillType.CUSTOM)
async def my_skill(param: str) -> Dict[str, Any]:
    """Detailed docstring"""
    return {"result": param}

# ❌ Bad: بدون type hints
def my_skill(param):
    return param
```

### 10.2 Error Handling در Skills

```python
# ✅ Good: با try-except
async def my_skill(param: str):
    try:
        result = do_something(param)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Skill failed: {e}")
        return {"success": False, "error": str(e)}

# ❌ Bad: بدون error handling
async def my_skill(param: str):
    result = do_something(param)  # ممکن است exception بدهد
    return result
```

### 10.3 Dependencies در Steps

```python
# ✅ Good: dependencies واضح
steps = [
    {"id": 1, "action": "retrieve", "depends_on": []},
    {"id": 2, "action": "reason", "depends_on": [1]},
    {"id": 3, "action": "validate", "depends_on": [2]}
]

# ❌ Bad: circular dependency
steps = [
    {"id": 1, "action": "retrieve", "depends_on": [2]},
    {"id": 2, "action": "reason", "depends_on": [1]}
]
```

---

## 11. نتیجه‌گیری

### 11.1 ویژگی‌های کلیدی

1. ✅ **Extensible**: امکان اضافه کردن skills جدید
2. ✅ **Type-safe**: با type hints و validation
3. ✅ **Error-resilient**: با error handling جامع
4. ✅ **Observable**: با logging و monitoring
5. ✅ **Flexible**: پشتیبانی از actions و tools مختلف

### 11.2 مزایا

- **یکپارچگی کامل**: با phi4 و orchestrator
- **قابلیت اطمینان**: با error handling و fallback
- **عملکرد**: با اجرای موازی و ترتیبی
- **قابلیت توسعه**: با registry و decorator pattern

### 11.3 استفاده در Production

```python
# Setup
orchestrator = AdaptiveOrchestrator(config)

# Register custom skills
@register_skill("custom_api", "Call external API", {...})
async def custom_api(endpoint: str):
    # Implementation
    pass

# Run query
state = await orchestrator.run(query, session_id)

# Check results
if state.tool_results:
    for result in state.tool_results:
        print(f"Tool: {result['tool']}, Success: {result['success']}")

if state.step_results:
    for step_id, result in state.step_results.items():
        print(f"Step {step_id}: {result['success']}")
```

---

## پیوست: API Reference

### SkillRegistry

```python
class SkillRegistry:
    def register(name, func, description, parameters, skill_type)
    def get(name) -> Callable
    def list_skills() -> List[Dict]
    def has_skill(name) -> bool
```

### SkillExecutor

```python
class SkillExecutor:
    async def execute_skill(skill_name, args, context) -> SkillResult
    async def execute_step_with_skill(step, context) -> SkillResult
    async def execute_multiple_skills(skill_calls, parallel) -> List[SkillResult]
    def get_execution_summary() -> Dict
```

### SkillResult

```python
@dataclass
class SkillResult:
    success: bool
    output: Any
    error: Optional[str]
    metadata: Optional[Dict]
    execution_time: float
```

**سیستم آماده استفاده است! 🚀**