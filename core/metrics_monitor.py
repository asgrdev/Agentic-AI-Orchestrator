"""
Metrics and Monitoring System
Tracks performance, usage, and system health
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    query: str
    session_id: str
    query_type: str
    complexity: str
    start_time: float
    end_time: Optional[float] = None
    total_time: Optional[float] = None
    steps_executed: List[str] = field(default_factory=list)
    step_timings: Dict[str, float] = field(default_factory=dict)
    skills_used: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    confidence: float = 0.0
    quality_score: float = 0.0
    tokens_used: int = 0
    memory_peak_mb: float = 0.0


@dataclass
class SystemMetrics:
    """Aggregated system metrics"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_response_time: float = 0.0
    average_confidence: float = 0.0
    average_quality_score: float = 0.0
    total_tokens_used: int = 0
    queries_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    queries_by_complexity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    skills_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    peak_memory_mb: float = 0.0


class MetricsMonitor:
    """
    Monitor and track system metrics
    """
    
    def __init__(self, log_dir: str = "logs/metrics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_queries: Dict[str, QueryMetrics] = {}
        self.completed_queries: List[QueryMetrics] = []
        self.system_metrics = SystemMetrics()
        
        self._start_time = time.time()
        
    def start_query(self, query: str, session_id: str, query_type: str = "unknown", 
                   complexity: str = "unknown") -> str:
        """Start tracking a new query"""
        query_id = f"{session_id}_{int(time.time() * 1000)}"
        
        metrics = QueryMetrics(
            query=query,
            session_id=session_id,
            query_type=query_type,
            complexity=complexity,
            start_time=time.time()
        )
        
        self.current_queries[query_id] = metrics
        logger.info(f"Started tracking query: {query_id}")
        
        return query_id
    
    def record_step(self, query_id: str, step_name: str, duration: float):
        """Record execution of a step"""
        if query_id in self.current_queries:
            metrics = self.current_queries[query_id]
            metrics.steps_executed.append(step_name)
            metrics.step_timings[step_name] = duration
            logger.debug(f"Recorded step {step_name}: {duration:.3f}s")
    
    def record_skill_usage(self, query_id: str, skill_name: str):
        """Record usage of a skill"""
        if query_id in self.current_queries:
            metrics = self.current_queries[query_id]
            metrics.skills_used.append(skill_name)
            
            # Update system metrics
            self.system_metrics.skills_usage[skill_name] += 1
            logger.debug(f"Recorded skill usage: {skill_name}")
    
    def end_query(self, query_id: str, success: bool = True, error: Optional[str] = None,
                 confidence: float = 0.0, quality_score: float = 0.0, 
                 tokens_used: int = 0, memory_peak_mb: float = 0.0):
        """End tracking a query and update metrics"""
        if query_id not in self.current_queries:
            logger.warning(f"Query {query_id} not found in current queries")
            return
        
        metrics = self.current_queries[query_id]
        metrics.end_time = time.time()
        metrics.total_time = metrics.end_time - metrics.start_time
        metrics.success = success
        metrics.error = error
        metrics.confidence = confidence
        metrics.quality_score = quality_score
        metrics.tokens_used = tokens_used
        metrics.memory_peak_mb = memory_peak_mb
        
        # Move to completed
        self.completed_queries.append(metrics)
        del self.current_queries[query_id]
        
        # Update system metrics
        self._update_system_metrics(metrics)
        
        logger.info(f"Completed query {query_id}: {metrics.total_time:.2f}s, "
                   f"success={success}, confidence={confidence:.2f}")
    
    def _update_system_metrics(self, query_metrics: QueryMetrics):
        """Update aggregated system metrics"""
        self.system_metrics.total_queries += 1
        
        if query_metrics.success:
            self.system_metrics.successful_queries += 1
        else:
            self.system_metrics.failed_queries += 1
            if query_metrics.error:
                error_type = type(query_metrics.error).__name__ if isinstance(query_metrics.error, Exception) else "Unknown"
                self.system_metrics.error_types[error_type] += 1
        
        # Update averages
        total = self.system_metrics.total_queries
        self.system_metrics.average_response_time = (
            (self.system_metrics.average_response_time * (total - 1) + query_metrics.total_time) / total
        )
        self.system_metrics.average_confidence = (
            (self.system_metrics.average_confidence * (total - 1) + query_metrics.confidence) / total
        )
        self.system_metrics.average_quality_score = (
            (self.system_metrics.average_quality_score * (total - 1) + query_metrics.quality_score) / total
        )
        
        # Update totals
        self.system_metrics.total_tokens_used += query_metrics.tokens_used
        self.system_metrics.queries_by_type[query_metrics.query_type] += 1
        self.system_metrics.queries_by_complexity[query_metrics.complexity] += 1
        
        if query_metrics.memory_peak_mb > self.system_metrics.peak_memory_mb:
            self.system_metrics.peak_memory_mb = query_metrics.memory_peak_mb
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        uptime = time.time() - self._start_time
        
        return {
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "total_queries": self.system_metrics.total_queries,
            "successful_queries": self.system_metrics.successful_queries,
            "failed_queries": self.system_metrics.failed_queries,
            "success_rate": (
                self.system_metrics.successful_queries / self.system_metrics.total_queries 
                if self.system_metrics.total_queries > 0 else 0.0
            ),
            "average_response_time": self.system_metrics.average_response_time,
            "average_confidence": self.system_metrics.average_confidence,
            "average_quality_score": self.system_metrics.average_quality_score,
            "total_tokens_used": self.system_metrics.total_tokens_used,
            "queries_by_type": dict(self.system_metrics.queries_by_type),
            "queries_by_complexity": dict(self.system_metrics.queries_by_complexity),
            "top_skills": self._get_top_skills(10),
            "error_types": dict(self.system_metrics.error_types),
            "peak_memory_mb": self.system_metrics.peak_memory_mb,
            "active_queries": len(self.current_queries),
        }
    
    def _get_top_skills(self, n: int = 10) -> List[tuple]:
        """Get top N most used skills"""
        sorted_skills = sorted(
            self.system_metrics.skills_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_skills[:n]
    
    def get_query_metrics(self, query_id: str) -> Optional[Dict]:
        """Get metrics for a specific query"""
        # Check current queries
        if query_id in self.current_queries:
            metrics = self.current_queries[query_id]
            return self._metrics_to_dict(metrics)
        
        # Check completed queries
        for metrics in self.completed_queries:
            if f"{metrics.session_id}_{int(metrics.start_time * 1000)}" == query_id:
                return self._metrics_to_dict(metrics)
        
        return None
    
    def _metrics_to_dict(self, metrics: QueryMetrics) -> Dict:
        """Convert QueryMetrics to dictionary"""
        return {
            "query": metrics.query,
            "session_id": metrics.session_id,
            "query_type": metrics.query_type,
            "complexity": metrics.complexity,
            "start_time": datetime.fromtimestamp(metrics.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(metrics.end_time).isoformat() if metrics.end_time else None,
            "total_time": metrics.total_time,
            "steps_executed": metrics.steps_executed,
            "step_timings": metrics.step_timings,
            "skills_used": metrics.skills_used,
            "success": metrics.success,
            "error": str(metrics.error) if metrics.error else None,
            "confidence": metrics.confidence,
            "quality_score": metrics.quality_score,
            "tokens_used": metrics.tokens_used,
            "memory_peak_mb": metrics.memory_peak_mb,
        }
    
    def export_metrics(self, filename: Optional[str] = None):
        """Export metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        data = {
            "system_metrics": self.get_system_metrics(),
            "completed_queries": [
                self._metrics_to_dict(m) for m in self.completed_queries[-100:]  # Last 100
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported metrics to {filepath}")
        return str(filepath)
    
    def print_summary(self):
        """Print a summary of system metrics"""
        metrics = self.get_system_metrics()
        
        print("\n" + "="*80)
        print("SYSTEM METRICS SUMMARY")
        print("="*80)
        
        print(f"\n📊 Overview:")
        print(f"  Uptime: {metrics['uptime_hours']:.2f} hours")
        print(f"  Total Queries: {metrics['total_queries']}")
        print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
        print(f"  Active Queries: {metrics['active_queries']}")
        
        print(f"\n⚡ Performance:")
        print(f"  Avg Response Time: {metrics['average_response_time']:.2f}s")
        print(f"  Avg Confidence: {metrics['average_confidence']:.2f}")
        print(f"  Avg Quality Score: {metrics['average_quality_score']:.2f}")
        print(f"  Peak Memory: {metrics['peak_memory_mb']:.1f} MB")
        
        print(f"\n📈 Query Distribution:")
        for qtype, count in metrics['queries_by_type'].items():
            percentage = (count / metrics['total_queries'] * 100) if metrics['total_queries'] > 0 else 0
            print(f"  {qtype}: {count} ({percentage:.1f}%)")
        
        print(f"\n🔧 Top Skills:")
        for skill, count in metrics['top_skills'][:5]:
            print(f"  {skill}: {count} uses")
        
        if metrics['error_types']:
            print(f"\n❌ Errors:")
            for error_type, count in metrics['error_types'].items():
                print(f"  {error_type}: {count}")
        
        print("\n" + "="*80 + "\n")


# Global metrics monitor instance
_metrics_monitor: Optional[MetricsMonitor] = None


def get_metrics_monitor() -> MetricsMonitor:
    """Get or create global metrics monitor"""
    global _metrics_monitor
    if _metrics_monitor is None:
        _metrics_monitor = MetricsMonitor()
    return _metrics_monitor


def reset_metrics_monitor():
    """Reset global metrics monitor"""
    global _metrics_monitor
    _metrics_monitor = None

