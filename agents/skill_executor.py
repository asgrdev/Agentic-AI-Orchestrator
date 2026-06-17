"""
Skill/Tool Executor for AdaptiveOrchestrator

این ماژول مسئول اجرای tool calls در steps است.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SkillType(Enum):
    """انواع skill ها"""
    SEARCH = "search"              # جستجو در منابع مختلف
    RETRIEVE = "retrieve"          # بازیابی از vector store
    GRAPH_QUERY = "graph_query"    # کوئری از knowledge graph
    WEB_SEARCH = "web_search"      # جستجوی وب
    CALCULATE = "calculate"        # محاسبات
    TRANSFORM = "transform"        # تبدیل داده
    VALIDATE = "validate"          # اعتبارسنجی
    SUMMARIZE = "summarize"        # خلاصه‌سازی
    COMPARE = "compare"            # مقایسه
    ANALYZE = "analyze"            # تحلیل


@dataclass
class SkillResult:
    """نتیجه اجرای یک skill"""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0


class SkillRegistry:
    """
    Registry برای ثبت و مدیریت skills/tools
    
    Skills می‌توانند:
    - توابع Python باشند
    - API calls باشند
    - External services باشند
    """
    
    def __init__(self):
        self._skills: Dict[str, Callable] = {}
        self._skill_metadata: Dict[str, Dict[str, Any]] = {}
        self._register_default_skills()
    
    def register(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        skill_type: Optional[SkillType] = None
    ):
        """
        ثبت یک skill جدید
        
        Args:
            name: نام skill
            func: تابع قابل فراخوانی
            description: توضیحات
            parameters: پارامترهای مورد نیاز
            skill_type: نوع skill
        """
        self._skills[name] = func
        self._skill_metadata[name] = {
            "description": description,
            "parameters": parameters or {},
            "type": skill_type.value if skill_type else "custom"
        }
        logger.info(f"Registered skill: {name}")
    
    def get(self, name: str) -> Optional[Callable]:
        """دریافت یک skill"""
        return self._skills.get(name)
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """لیست تمام skills"""
        return [
            {
                "name": name,
                **metadata
            }
            for name, metadata in self._skill_metadata.items()
        ]
    
    def has_skill(self, name: str) -> bool:
        """بررسی وجود skill"""
        return name in self._skills
    
    def _register_default_skills(self):
        """ثبت skills پیش‌فرض"""
        
        # Search skill
        async def search_skill(query: str, sources: List[str] = None) -> Dict[str, Any]:
            """جستجو در منابع مختلف"""
            logger.info(f"Executing search: {query}")
            return {
                "query": query,
                "sources": sources or ["vector", "graph"],
                "results": []  # باید با سیستم واقعی جایگزین شود
            }
        
        self.register(
            "search",
            search_skill,
            "Search across multiple sources",
            {"query": "string", "sources": "list[string]"},
            SkillType.SEARCH
        )
        
        # Web search skill
        async def web_search_skill(query: str, num_results: int = 5) -> Dict[str, Any]:
            """جستجوی وب"""
            logger.info(f"Executing web search: {query}")
            return {
                "query": query,
                "num_results": num_results,
                "results": []  # باید با API واقعی جایگزین شود
            }
        
        self.register(
            "web_search",
            web_search_skill,
            "Search the web for information",
            {"query": "string", "num_results": "int"},
            SkillType.WEB_SEARCH
        )
        
        # Graph query skill
        async def graph_query_skill(cypher: str) -> Dict[str, Any]:
            """کوئری از knowledge graph"""
            logger.info(f"Executing graph query: {cypher[:50]}...")
            return {
                "cypher": cypher,
                "results": []  # باید با graph client واقعی جایگزین شود
            }
        
        self.register(
            "graph_query",
            graph_query_skill,
            "Query the knowledge graph",
            {"cypher": "string"},
            SkillType.GRAPH_QUERY
        )
        
        # Calculate skill
        async def calculate_skill(expression: str) -> Dict[str, Any]:
            """محاسبات ریاضی"""
            logger.info(f"Calculating: {expression}")
            try:
                # ساده‌سازی شده - در production باید ایمن‌تر باشد
                result = eval(expression, {"__builtins__": {}}, {})
                return {"expression": expression, "result": result}
            except Exception as e:
                return {"expression": expression, "error": str(e)}
        
        self.register(
            "calculate",
            calculate_skill,
            "Perform mathematical calculations",
            {"expression": "string"},
            SkillType.CALCULATE
        )


class SkillExecutor:
    """
    اجراکننده skills در context یک plan
    
    مسئولیت‌ها:
    - اجرای tool calls
    - مدیریت dependencies
    - Error handling
    - Logging و monitoring
    """
    
    def __init__(self, registry: Optional[SkillRegistry] = None):
        self.registry = registry or SkillRegistry()
        self.execution_history: List[Dict[str, Any]] = []
    
    async def execute_skill(
        self,
        skill_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SkillResult:
        """
        اجرای یک skill
        
        Args:
            skill_name: نام skill
            args: آرگومان‌های skill
            context: context اضافی
            
        Returns:
            SkillResult با نتیجه اجرا
        """
        import time
        from asyncio import iscoroutinefunction
        
        start_time = time.time()
        
        try:
            # بررسی وجود skill
            if not self.registry.has_skill(skill_name):
                logger.error(f"Skill not found: {skill_name}")
                return SkillResult(
                    success=False,
                    output=None,
                    error=f"Skill '{skill_name}' not found",
                    execution_time=time.time() - start_time
                )
            
            # دریافت skill
            skill_func = self.registry.get(skill_name)
            
            # اجرا
            logger.info(f"Executing skill: {skill_name} with args: {args}")
            
            if iscoroutinefunction(skill_func):
                output = await skill_func(**args)
            else:
                output = skill_func(**args)
            
            execution_time = time.time() - start_time
            
            # ثبت در history
            self.execution_history.append({
                "skill": skill_name,
                "args": args,
                "success": True,
                "execution_time": execution_time
            })
            
            logger.info(f"Skill executed successfully: {skill_name} in {execution_time:.2f}s")
            
            return SkillResult(
                success=True,
                output=output,
                execution_time=execution_time,
                metadata={"skill": skill_name, "args": args}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Skill execution failed: {skill_name} - {str(e)}", exc_info=True)
            
            # ثبت خطا در history
            self.execution_history.append({
                "skill": skill_name,
                "args": args,
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            })
            
            return SkillResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time,
                metadata={"skill": skill_name, "args": args}
            )
    
    async def execute_step_with_skill(
        self,
        step: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SkillResult:
        """
        اجرای یک step که شامل skill call است
        
        Args:
            step: step dictionary با tool و args
            context: context اضافی
            
        Returns:
            SkillResult
        """
        tool_name = step.get("tool")
        args = step.get("args", {})
        
        if not tool_name:
            logger.warning(f"Step {step.get('id')} has no tool specified")
            return SkillResult(
                success=False,
                output=None,
                error="No tool specified in step"
            )
        
        return await self.execute_skill(tool_name, args, context)
    
    async def execute_multiple_skills(
        self,
        skill_calls: List[Dict[str, Any]],
        parallel: bool = False
    ) -> List[SkillResult]:
        """
        اجرای چندین skill
        
        Args:
            skill_calls: لیست skill calls با format {"tool": "name", "args": {...}}
            parallel: اجرای موازی یا ترتیبی
            
        Returns:
            لیست SkillResult
        """
        if parallel:
            import asyncio
            tasks = [
                self.execute_skill(call["tool"], call.get("args", {}))
                for call in skill_calls
            ]
            raw = await asyncio.gather(*tasks, return_exceptions=True)
            return [
                r if isinstance(r, SkillResult)
                else SkillResult(success=False, output=None, error=str(r))
                for r in raw
            ]
        else:
            results = []
            for call in skill_calls:
                result = await self.execute_skill(call["tool"], call.get("args", {}))
                results.append(result)
            return results
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """خلاصه اجراهای انجام شده"""
        total = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h.get("success", False))
        failed = total - successful
        
        total_time = sum(h.get("execution_time", 0) for h in self.execution_history)
        avg_time = total_time / total if total > 0 else 0
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "total_time": total_time,
            "average_time": avg_time,
            "history": self.execution_history
        }


# Global registry instance
_global_registry = SkillRegistry()


def get_global_registry() -> SkillRegistry:
    """دریافت global registry"""
    return _global_registry


def register_skill(
    name: str,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    skill_type: Optional[SkillType] = None
):
    """
    Decorator برای ثبت skill
    
    Usage:
        @register_skill("my_skill", "Does something", {"param": "string"})
        async def my_skill(param: str):
            return {"result": param}
    """
    def decorator(func: Callable):
        _global_registry.register(name, func, description, parameters, skill_type)
        return func
    return decorator

