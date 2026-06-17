"""
Visualization API - Backend for Advanced UI

این API endpoint های لازم برای UI پیشرفته را فراهم می‌کند:
- Sensor monitoring
- Knowledge graph visualization
- Agent flow visualization
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# Pydantic Models
# ══════════════════════════════════════════════════════════════

class SensorStatus(BaseModel):
    """وضعیت یک سنسور"""
    sensor_type: str  # vision, audio, text
    status: str  # active, idle, processing, error
    last_input: Optional[str] = None
    last_output: Optional[dict] = None
    processing_time: float = 0.0
    model_name: str = ""
    confidence: float = 0.0
    timestamp: str = ""


class GraphNode(BaseModel):
    """یک node در گراف دانش"""
    id: str
    name: str
    type: str
    canonical_name: str = ""
    confidence: float = 0.0
    version: int = 1
    description: str = ""


class GraphEdge(BaseModel):
    """یک edge در گراف دانش"""
    from_id: str
    to_id: str
    rel_type: str
    relation_type: str = "explicit"  # explicit, inferred, semantic
    confidence: float = 0.0
    weight: float = 0.0


class GraphStats(BaseModel):
    """آمار گراف دانش"""
    entity_count: int
    relation_count: int
    explicit_relations: int
    inferred_relations: int
    semantic_relations: int
    quality_score: float
    avg_confidence: float
    density: float


class AgentFlowNode(BaseModel):
    """یک node در فلوی agent"""
    id: str
    type: str  # start, agent, tool, decision, end
    label: str
    status: str  # pending, active, completed, error
    metadata: dict[str, Any] = {}
    timestamp: str = ""


class AgentFlowEdge(BaseModel):
    """یک edge در فلوی agent"""
    from_id: str
    to_id: str
    label: str = ""
    condition: str = ""


class AgentFlowState(BaseModel):
    """وضعیت کامل فلوی agent"""
    nodes: list[AgentFlowNode]
    edges: list[AgentFlowEdge]
    current_node: str
    query: str
    step_count: int
    duration: float


# ══════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Agentic Graph RAG Visualization API",
    description="API for advanced visualization of sensors, knowledge graph, and agent flow",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در production باید محدود شود
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
# Global State (در production باید از Redis استفاده شود)
# ══════════════════════════════════════════════════════════════

sensor_states: dict[str, SensorStatus] = {}
graph_cache: dict[str, Any] = {}
agent_flow_cache: dict[str, AgentFlowState] = {}
websocket_connections: list[WebSocket] = []


# ══════════════════════════════════════════════════════════════
# Sensor Monitoring Endpoints
# ══════════════════════════════════════════════════════════════

@app.get("/api/sensors/status", response_model=list[SensorStatus])
async def get_sensors_status():
    """دریافت وضعیت تمام سنسورها"""
    return list(sensor_states.values())


@app.get("/api/sensors/{sensor_type}", response_model=SensorStatus)
async def get_sensor_status(sensor_type: str):
    """دریافت وضعیت یک سنسور خاص"""
    if sensor_type not in sensor_states:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor_type} not found")
    return sensor_states[sensor_type]


@app.post("/api/sensors/{sensor_type}/process")
async def process_sensor_input(sensor_type: str, input_data: dict):
    """پردازش ورودی یک سنسور"""
    try:
        # Update sensor status
        sensor_states[sensor_type] = SensorStatus(
            sensor_type=sensor_type,
            status="processing",
            last_input=str(input_data.get("input", "")),
            timestamp=datetime.now().isoformat(),
        )
        
        # Broadcast to WebSocket clients
        await broadcast_sensor_update(sensor_type)
        
        # Simulate processing (در واقعیت باید sensor را فراخوانی کند)
        await asyncio.sleep(0.5)
        
        # Update with result
        sensor_states[sensor_type].status = "active"
        sensor_states[sensor_type].last_output = {"result": "processed"}
        sensor_states[sensor_type].processing_time = 0.5
        
        await broadcast_sensor_update(sensor_type)
        
        return {"status": "success", "sensor": sensor_type}
    
    except Exception as e:
        logger.error(f"Sensor processing failed: {e}")
        sensor_states[sensor_type].status = "error"
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# Knowledge Graph Endpoints
# ══════════════════════════════════════════════════════════════

@app.get("/api/graph/stats", response_model=GraphStats)
async def get_graph_stats():
    """دریافت آمار گراف دانش"""
    # در واقعیت باید از KuzuDB بخواند
    return GraphStats(
        entity_count=graph_cache.get("entity_count", 0),
        relation_count=graph_cache.get("relation_count", 0),
        explicit_relations=graph_cache.get("explicit_relations", 0),
        inferred_relations=graph_cache.get("inferred_relations", 0),
        semantic_relations=graph_cache.get("semantic_relations", 0),
        quality_score=graph_cache.get("quality_score", 0.0),
        avg_confidence=graph_cache.get("avg_confidence", 0.0),
        density=graph_cache.get("density", 0.0),
    )


@app.get("/api/graph/nodes", response_model=list[GraphNode])
async def get_graph_nodes(limit: int = 100, offset: int = 0):
    """دریافت nodes گراف دانش"""
    # در واقعیت باید از KuzuDB بخواند
    nodes = graph_cache.get("nodes", [])
    return nodes[offset:offset + limit]


@app.get("/api/graph/edges", response_model=list[GraphEdge])
async def get_graph_edges(limit: int = 100, offset: int = 0):
    """دریافت edges گراف دانش"""
    # در واقعیت باید از KuzuDB بخواند
    edges = graph_cache.get("edges", [])
    return edges[offset:offset + limit]


@app.get("/api/graph/subgraph")
async def get_subgraph(entity_id: str, depth: int = 2):
    """دریافت subgraph اطراف یک entity"""
    # در واقعیت باید از KuzuDB بخواند
    return {
        "center": entity_id,
        "nodes": [],
        "edges": [],
        "depth": depth,
    }


@app.get("/api/graph/search")
async def search_graph(query: str, limit: int = 20):
    """جستجو در گراف دانش"""
    # در واقعیت باید از KuzuDB بخواند
    return {
        "query": query,
        "results": [],
    }


# ══════════════════════════════════════════════════════════════
# Agent Flow Endpoints
# ══════════════════════════════════════════════════════════════

@app.get("/api/flow/current", response_model=AgentFlowState)
async def get_current_flow():
    """دریافت فلوی فعلی agent"""
    if "current" not in agent_flow_cache:
        raise HTTPException(status_code=404, detail="No active flow")
    return agent_flow_cache["current"]


@app.get("/api/flow/history")
async def get_flow_history(limit: int = 10):
    """دریافت تاریخچه فلوها"""
    history = agent_flow_cache.get("history", [])
    return history[-limit:]


@app.post("/api/flow/start")
async def start_flow(query: str):
    """شروع یک فلوی جدید"""
    flow = AgentFlowState(
        nodes=[
            AgentFlowNode(
                id="start",
                type="start",
                label="Start",
                status="completed",
                timestamp=datetime.now().isoformat(),
            ),
            AgentFlowNode(
                id="understand",
                type="agent",
                label="Understand & Plan",
                status="active",
                timestamp=datetime.now().isoformat(),
            ),
        ],
        edges=[
            AgentFlowEdge(from_id="start", to_id="understand", label="query"),
        ],
        current_node="understand",
        query=query,
        step_count=1,
        duration=0.0,
    )
    
    agent_flow_cache["current"] = flow
    await broadcast_flow_update()
    
    return {"status": "started", "flow_id": "current"}


@app.post("/api/flow/step")
async def flow_step(node_id: str, status: str, metadata: dict = {}):
    """بروزرسانی یک step در فلو"""
    if "current" not in agent_flow_cache:
        raise HTTPException(status_code=404, detail="No active flow")
    
    flow = agent_flow_cache["current"]
    
    # Update node status
    for node in flow.nodes:
        if node.id == node_id:
            node.status = status
            node.metadata = metadata
            node.timestamp = datetime.now().isoformat()
            break
    
    flow.step_count += 1
    await broadcast_flow_update()
    
    return {"status": "updated", "node": node_id}


# ══════════════════════════════════════════════════════════════
# WebSocket for Real-time Updates
# ══════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket برای بروزرسانی real-time"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Echo back (برای heartbeat)
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        logger.info("WebSocket client disconnected")


async def broadcast_sensor_update(sensor_type: str):
    """ارسال بروزرسانی سنسور به تمام clients"""
    if not websocket_connections:
        return
    
    message = {
        "type": "sensor_update",
        "sensor": sensor_type,
        "data": sensor_states[sensor_type].dict(),
    }
    
    for ws in websocket_connections:
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.error(f"WebSocket send failed: {e}")


async def broadcast_flow_update():
    """ارسال بروزرسانی فلو به تمام clients"""
    if not websocket_connections or "current" not in agent_flow_cache:
        return
    
    message = {
        "type": "flow_update",
        "data": agent_flow_cache["current"].dict(),
    }
    
    for ws in websocket_connections:
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.error(f"WebSocket send failed: {e}")


# ══════════════════════════════════════════════════════════════
# Utility Endpoints
# ══════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """بررسی سلامت API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "sensors": len(sensor_states),
        "websockets": len(websocket_connections),
    }


@app.post("/api/reset")
async def reset_state():
    """ریست کردن state (برای تست)"""
    sensor_states.clear()
    graph_cache.clear()
    agent_flow_cache.clear()
    return {"status": "reset"}


# ══════════════════════════════════════════════════════════════
# Startup & Shutdown
# ══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """راه‌اندازی API"""
    logger.info("🚀 Visualization API started")
    
    # Initialize default sensor states
    for sensor_type in ["vision", "audio", "text"]:
        sensor_states[sensor_type] = SensorStatus(
            sensor_type=sensor_type,
            status="idle",
            timestamp=datetime.now().isoformat(),
        )


@app.on_event("shutdown")
async def shutdown_event():
    """خاموش کردن API"""
    logger.info("👋 Visualization API shutdown")
    
    # Close all WebSocket connections
    for ws in websocket_connections:
        await ws.close()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "visualization_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


