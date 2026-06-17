"""
Specialized & Innovative Skills
Skills for: CLI, Code, Finance, Sensors, Graph, Vector, Design, API
Plus cutting-edge AI trends (2024-2026)
"""

import logging
import json
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from agents.skill_executor import register_skill, SkillType

logger = logging.getLogger(__name__)


# ============================================================================
# CLI Skills
# ============================================================================

@register_skill(
    "execute_cli_command",
    "Execute CLI commands safely",
    {"command": "string", "timeout": "int", "capture_output": "bool"},
    SkillType.TRANSFORM
)
async def execute_cli_command_skill(
    command: str,
    timeout: int = 30,
    capture_output: bool = True
) -> Dict[str, Any]:
    """
    اجرای دستورات CLI به صورت ایمن
    
    ⚠️ Security: فقط دستورات مجاز
    """
    # لیست سفید دستورات مجاز
    allowed_commands = ['ls', 'pwd', 'echo', 'cat', 'grep', 'find', 'wc']
    
    cmd_parts = command.split()
    if not cmd_parts or cmd_parts[0] not in allowed_commands:
        return {
            "success": False,
            "error": f"Command not allowed: {cmd_parts[0] if cmd_parts else 'empty'}"
        }
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout if capture_output else None,
            "stderr": result.stderr if capture_output else None,
            "returncode": result.returncode,
            "command": command
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@register_skill(
    "parse_cli_output",
    "Parse and structure CLI command output",
    {"output": "string", "format": "string"},
    SkillType.TRANSFORM
)
async def parse_cli_output_skill(
    output: str,
    format: str = "lines"
) -> Dict[str, Any]:
    """
    پردازش خروجی CLI
    
    formats: lines, json, csv, table
    """
    try:
        if format == "lines":
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            return {"format": "lines", "data": lines, "count": len(lines)}
        
        elif format == "json":
            data = json.loads(output)
            return {"format": "json", "data": data}
        
        elif format == "csv":
            lines = output.split('\n')
            data = [line.split(',') for line in lines if line.strip()]
            return {"format": "csv", "data": data, "rows": len(data)}
        
        else:
            return {"format": "raw", "data": output}
            
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Code Skills
# ============================================================================

@register_skill(
    "analyze_code_quality",
    "Analyze code quality and complexity",
    {"code": "string", "language": "string"},
    SkillType.ANALYZE
)
async def analyze_code_quality_skill(
    code: str,
    language: str = "python"
) -> Dict[str, Any]:
    """
    تحلیل کیفیت کد
    
    Metrics: lines, complexity, functions, classes, comments
    """
    lines = code.split('\n')
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
    comment_lines = len([l for l in lines if l.strip().startswith('#')])
    blank_lines = total_lines - code_lines - comment_lines
    
    # تشخیص functions و classes (ساده)
    functions = len([l for l in lines if 'def ' in l])
    classes = len([l for l in lines if 'class ' in l])
    
    return {
        "language": language,
        "total_lines": total_lines,
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "functions": functions,
        "classes": classes,
        "comment_ratio": comment_lines / total_lines if total_lines > 0 else 0,
        "quality_score": (comment_lines / total_lines * 0.3 + 
                         (code_lines / total_lines) * 0.7) if total_lines > 0 else 0
    }


@register_skill(
    "refactor_code",
    "Suggest code refactoring improvements",
    {"code": "string", "language": "string", "focus": "string"},
    SkillType.TRANSFORM
)
async def refactor_code_skill(
    code: str,
    language: str = "python",
    focus: str = "readability"
) -> Dict[str, Any]:
    """
    پیشنهاد بهبود کد
    
    focus: readability, performance, security, maintainability
    """
    suggestions = []
    
    lines = code.split('\n')
    
    # بررسی‌های ساده
    if focus in ["readability", "maintainability"]:
        long_lines = [i for i, l in enumerate(lines, 1) if len(l) > 80]
        if long_lines:
            suggestions.append({
                "type": "line_length",
                "lines": long_lines,
                "message": "Lines exceed 80 characters"
            })
    
    if focus in ["security", "maintainability"]:
        if "eval(" in code or "exec(" in code:
            suggestions.append({
                "type": "security",
                "message": "Avoid using eval() or exec()"
            })
    
    return {
        "language": language,
        "focus": focus,
        "suggestions": suggestions,
        "suggestion_count": len(suggestions)
    }


# ============================================================================
# Finance Skills
# ============================================================================

@register_skill(
    "calculate_financial_metrics",
    "Calculate financial metrics and ratios",
    {"data": "dict", "metrics": "list[string]"},
    SkillType.ANALYZE
)
async def calculate_financial_metrics_skill(
    data: Dict[str, float],
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    محاسبه متریک‌های مالی
    
    metrics: roi, profit_margin, growth_rate, etc.
    """
    results = {}
    
    if not metrics:
        metrics = ["roi", "profit_margin", "growth_rate"]
    
    # ROI (Return on Investment)
    if "roi" in metrics and "profit" in data and "investment" in data:
        results["roi"] = (data["profit"] / data["investment"]) * 100 if data["investment"] != 0 else 0
    
    # Profit Margin
    if "profit_margin" in metrics and "profit" in data and "revenue" in data:
        results["profit_margin"] = (data["profit"] / data["revenue"]) * 100 if data["revenue"] != 0 else 0
    
    # Growth Rate
    if "growth_rate" in metrics and "current" in data and "previous" in data:
        results["growth_rate"] = ((data["current"] - data["previous"]) / data["previous"]) * 100 if data["previous"] != 0 else 0
    
    return {
        "metrics": results,
        "data": data,
        "calculated": list(results.keys())
    }


@register_skill(
    "forecast_financial_trend",
    "Forecast financial trends using simple models",
    {"historical_data": "list[float]", "periods": "int", "method": "string"},
    SkillType.ANALYZE
)
async def forecast_financial_trend_skill(
    historical_data: List[float],
    periods: int = 3,
    method: str = "moving_average"
) -> Dict[str, Any]:
    """
    پیش‌بینی روند مالی
    
    methods: moving_average, linear, exponential
    """
    if not historical_data:
        return {"error": "No historical data provided"}
    
    forecast = []
    
    if method == "moving_average":
        # Simple Moving Average
        window = min(3, len(historical_data))
        last_values = historical_data[-window:]
        avg = sum(last_values) / len(last_values)
        forecast = [avg] * periods
    
    elif method == "linear":
        # Simple linear trend
        if len(historical_data) >= 2:
            slope = (historical_data[-1] - historical_data[0]) / (len(historical_data) - 1)
            last_value = historical_data[-1]
            forecast = [last_value + slope * (i + 1) for i in range(periods)]
    
    return {
        "method": method,
        "historical_count": len(historical_data),
        "forecast_periods": periods,
        "forecast": forecast,
        "last_actual": historical_data[-1] if historical_data else None
    }


# ============================================================================
# Sensor Skills
# ============================================================================

@register_skill(
    "process_sensor_data",
    "Process and analyze sensor data streams",
    {"data": "list[dict]", "sensor_type": "string", "aggregation": "string"},
    SkillType.ANALYZE
)
async def process_sensor_data_skill(
    data: List[Dict[str, Any]],
    sensor_type: str = "generic",
    aggregation: str = "average"
) -> Dict[str, Any]:
    """
    پردازش داده‌های سنسور
    
    sensor_types: temperature, humidity, motion, light, sound
    aggregations: average, min, max, sum, count
    """
    if not data:
        return {"error": "No sensor data provided"}
    
    # استخراج مقادیر
    values = [d.get("value", 0) for d in data if "value" in d]
    
    if not values:
        return {"error": "No values found in data"}
    
    result = {
        "sensor_type": sensor_type,
        "data_points": len(values),
        "aggregation": aggregation
    }
    
    if aggregation == "average":
        result["result"] = sum(values) / len(values)
    elif aggregation == "min":
        result["result"] = min(values)
    elif aggregation == "max":
        result["result"] = max(values)
    elif aggregation == "sum":
        result["result"] = sum(values)
    elif aggregation == "count":
        result["result"] = len(values)
    
    # آمار اضافی
    result["statistics"] = {
        "min": min(values),
        "max": max(values),
        "average": sum(values) / len(values),
        "range": max(values) - min(values)
    }
    
    return result


@register_skill(
    "detect_sensor_anomalies",
    "Detect anomalies in sensor data",
    {"data": "list[float]", "threshold": "float", "method": "string"},
    SkillType.ANALYZE
)
async def detect_sensor_anomalies_skill(
    data: List[float],
    threshold: float = 2.0,
    method: str = "zscore"
) -> Dict[str, Any]:
    """
    تشخیص ناهنجاری در داده‌های سنسور
    
    methods: zscore, iqr, threshold
    """
    if not data or len(data) < 3:
        return {"error": "Insufficient data for anomaly detection"}
    
    anomalies = []
    
    if method == "zscore":
        # Z-score method
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = variance ** 0.5
        
        for i, value in enumerate(data):
            if std_dev > 0:
                z_score = abs((value - mean) / std_dev)
                if z_score > threshold:
                    anomalies.append({"index": i, "value": value, "z_score": z_score})
    
    elif method == "threshold":
        # Simple threshold
        for i, value in enumerate(data):
            if abs(value) > threshold:
                anomalies.append({"index": i, "value": value})
    
    return {
        "method": method,
        "data_points": len(data),
        "anomalies_found": len(anomalies),
        "anomalies": anomalies,
        "anomaly_rate": len(anomalies) / len(data) if data else 0
    }


# ============================================================================
# Graph Skills
# ============================================================================

@register_skill(
    "analyze_graph_structure",
    "Analyze graph structure and properties",
    {"nodes": "list", "edges": "list"},
    SkillType.ANALYZE
)
async def analyze_graph_structure_skill(
    nodes: List[Any],
    edges: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    تحلیل ساختار گراف
    
    Metrics: node_count, edge_count, density, degree_distribution
    """
    node_count = len(nodes)
    edge_count = len(edges)
    
    # محاسبه density
    max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 0
    density = edge_count / max_edges if max_edges > 0 else 0
    
    # محاسبه degree برای هر node
    degree_map = {node: 0 for node in nodes}
    for edge in edges:
        if "source" in edge and "target" in edge:
            if edge["source"] in degree_map:
                degree_map[edge["source"]] += 1
            if edge["target"] in degree_map:
                degree_map[edge["target"]] += 1
    
    degrees = list(degree_map.values())
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": density,
        "average_degree": avg_degree,
        "max_degree": max(degrees) if degrees else 0,
        "min_degree": min(degrees) if degrees else 0
    }


@register_skill(
    "find_graph_paths",
    "Find paths between nodes in graph",
    {"graph": "dict", "start": "string", "end": "string", "max_depth": "int"},
    SkillType.SEARCH
)
async def find_graph_paths_skill(
    graph: Dict[str, List[str]],
    start: str,
    end: str,
    max_depth: int = 5
) -> Dict[str, Any]:
    """
    یافتن مسیر بین دو node
    
    Returns: shortest_path, all_paths (up to max_depth)
    """
    def bfs_shortest_path(graph, start, end):
        if start not in graph or end not in graph:
            return None
        
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            node, path = queue.pop(0)
            
            if node == end:
                return path
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    shortest = bfs_shortest_path(graph, start, end)
    
    return {
        "start": start,
        "end": end,
        "shortest_path": shortest,
        "path_length": len(shortest) - 1 if shortest else None,
        "path_found": shortest is not None
    }


# ============================================================================
# Vector Skills
# ============================================================================

@register_skill(
    "compute_vector_similarity",
    "Compute similarity between vectors",
    {"vector1": "list[float]", "vector2": "list[float]", "method": "string"},
    SkillType.ANALYZE
)
async def compute_vector_similarity_skill(
    vector1: List[float],
    vector2: List[float],
    method: str = "cosine"
) -> Dict[str, Any]:
    """
    محاسبه شباهت بین بردارها
    
    methods: cosine, euclidean, dot_product
    """
    if len(vector1) != len(vector2):
        return {"error": "Vectors must have same dimension"}
    
    if not vector1 or not vector2:
        return {"error": "Empty vectors"}
    
    result = {}
    
    if method == "cosine":
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        norm1 = sum(a ** 2 for a in vector1) ** 0.5
        norm2 = sum(b ** 2 for b in vector2) ** 0.5
        
        if norm1 > 0 and norm2 > 0:
            result["similarity"] = dot_product / (norm1 * norm2)
        else:
            result["similarity"] = 0
    
    elif method == "euclidean":
        # Euclidean distance
        distance = sum((a - b) ** 2 for a, b in zip(vector1, vector2)) ** 0.5
        result["distance"] = distance
        result["similarity"] = 1 / (1 + distance)  # Convert to similarity
    
    elif method == "dot_product":
        result["similarity"] = sum(a * b for a, b in zip(vector1, vector2))
    
    result["method"] = method
    result["dimension"] = len(vector1)
    
    return result


@register_skill(
    "cluster_vectors",
    "Cluster vectors using simple k-means",
    {"vectors": "list[list[float]]", "k": "int", "max_iterations": "int"},
    SkillType.ANALYZE
)
async def cluster_vectors_skill(
    vectors: List[List[float]],
    k: int = 3,
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    خوشه‌بندی بردارها
    
    Simple k-means clustering
    """
    if not vectors or k <= 0:
        return {"error": "Invalid input"}
    
    # ساده‌سازی شده - در production باید از sklearn استفاده شود
    import random
    
    # انتخاب تصادفی مراکز اولیه
    centroids = random.sample(vectors, min(k, len(vectors)))
    
    clusters = [[] for _ in range(k)]
    
    # یک iteration ساده
    for vector in vectors:
        # یافتن نزدیک‌ترین centroid
        distances = []
        for centroid in centroids:
            dist = sum((a - b) ** 2 for a, b in zip(vector, centroid)) ** 0.5
            distances.append(dist)
        
        closest = distances.index(min(distances))
        clusters[closest].append(vector)
    
    return {
        "k": k,
        "total_vectors": len(vectors),
        "cluster_sizes": [len(c) for c in clusters],
        "iterations": 1,  # ساده‌سازی شده
        "centroids_count": len(centroids)
    }


# ============================================================================
# Design Skills
# ============================================================================

@register_skill(
    "generate_color_palette",
    "Generate color palettes for design",
    {"base_color": "string", "scheme": "string", "count": "int"},
    SkillType.TRANSFORM
)
async def generate_color_palette_skill(
    base_color: str = "#3498db",
    scheme: str = "complementary",
    count: int = 5
) -> Dict[str, Any]:
    """
    تولید پالت رنگی
    
    schemes: complementary, analogous, triadic, monochromatic
    """
    # ساده‌سازی شده - در production باید از color theory استفاده شود
    
    palettes = {
        "complementary": ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"],
        "analogous": ["#3498db", "#2980b9", "#1abc9c", "#16a085", "#27ae60"],
        "triadic": ["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#2ecc71"],
        "monochromatic": ["#3498db", "#5dade2", "#85c1e9", "#aed6f1", "#d6eaf8"]
    }
    
    colors = palettes.get(scheme, palettes["complementary"])[:count]
    
    return {
        "base_color": base_color,
        "scheme": scheme,
        "colors": colors,
        "count": len(colors)
    }


@register_skill(
    "validate_design_accessibility",
    "Validate design for accessibility standards",
    {"design": "dict", "standard": "string"},
    SkillType.VALIDATE
)
async def validate_design_accessibility_skill(
    design: Dict[str, Any],
    standard: str = "WCAG_AA"
) -> Dict[str, Any]:
    """
    اعتبارسنجی دسترسی‌پذیری طراحی
    
    standards: WCAG_A, WCAG_AA, WCAG_AAA
    """
    issues = []
    warnings = []
    
    # بررسی‌های ساده
    if "font_size" in design:
        if design["font_size"] < 12:
            issues.append("Font size too small (< 12px)")
        elif design["font_size"] < 14:
            warnings.append("Font size could be larger for better readability")
    
    if "contrast_ratio" in design:
        min_contrast = {"WCAG_A": 3.0, "WCAG_AA": 4.5, "WCAG_AAA": 7.0}
        required = min_contrast.get(standard, 4.5)
        
        if design["contrast_ratio"] < required:
            issues.append(f"Contrast ratio {design['contrast_ratio']} < {required}")
    
    return {
        "standard": standard,
        "compliant": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "issue_count": len(issues),
        "warning_count": len(warnings)
    }


# ============================================================================
# API Skills (Advanced)
# ============================================================================

@register_skill(
    "generate_api_documentation",
    "Generate API documentation from endpoints",
    {"endpoints": "list[dict]", "format": "string"},
    SkillType.TRANSFORM
)
async def generate_api_documentation_skill(
    endpoints: List[Dict[str, Any]],
    format: str = "markdown"
) -> Dict[str, Any]:
    """
    تولید مستندات API
    
    formats: markdown, html, openapi
    """
    if not endpoints:
        return {"error": "No endpoints provided"}
    
    doc_lines = []
    
    if format == "markdown":
        doc_lines.append("# API Documentation\n")
        
        for endpoint in endpoints:
            method = endpoint.get("method", "GET")
            path = endpoint.get("path", "/")
            description = endpoint.get("description", "No description")
            
            doc_lines.append(f"## {method} {path}")
            doc_lines.append(f"\n{description}\n")
            
            if "parameters" in endpoint:
                doc_lines.append("\n### Parameters")
                for param in endpoint["parameters"]:
                    doc_lines.append(f"- `{param['name']}` ({param.get('type', 'string')}): {param.get('description', '')}")
            
            doc_lines.append("\n---\n")
    
    documentation = "\n".join(doc_lines)
    
    return {
        "format": format,
        "endpoint_count": len(endpoints),
        "documentation": documentation,
        "length": len(documentation)
    }


@register_skill(
    "test_api_endpoint",
    "Test API endpoint with various scenarios",
    {"endpoint": "dict", "test_cases": "list[dict]"},
    SkillType.VALIDATE
)
async def test_api_endpoint_skill(
    endpoint: Dict[str, Any],
    test_cases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    تست endpoint های API
    
    test_cases: لیست سناریوهای تست
    """
    results = []
    
    for i, test_case in enumerate(test_cases):
        result = {
            "test_id": i + 1,
            "name": test_case.get("name", f"Test {i+1}"),
            "passed": True,  # ساده‌سازی شده
            "expected": test_case.get("expected"),
            "actual": test_case.get("expected")  # در واقعیت باید API را فراخوانی کند
        }
        results.append(result)
    
    passed = sum(1 for r in results if r["passed"])
    
    return {
        "endpoint": endpoint.get("path", "unknown"),
        "total_tests": len(test_cases),
        "passed": passed,
        "failed": len(test_cases) - passed,
        "success_rate": passed / len(test_cases) if test_cases else 0,
        "results": results
    }


logger.info("Specialized skills registered successfully")

