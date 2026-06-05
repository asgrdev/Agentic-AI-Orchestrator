from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


from dataclasses import dataclass
 

# ─────────────────────────────────────────────
# Graph Models
# ─────────────────────────────────────────────
@dataclass
class GraphNode:
    id: str
    name: str
    canonical_name: str
    type: str
    description: str = ""
    confidence: float = 1.0
    linked_id: str = ""
    kb_url: str = ""
    embedding: Optional[List[float]] = None
    aliases: Optional[List[str]] = None


@dataclass
class GraphEdge:
    from_id: str
    to_id: str
    rel_type: str
    weight: float = 1.0
    confidence: float = 1.0
    source_doc: str = ""
    source_chunk: str = ""


@dataclass
class GraphDocument:
    id: str
    title: str
    source: str


@dataclass
class GraphChunk:
    id: str
    content: str
    doc_id: str
    chunk_idx: int


@dataclass
class ColumnDef:
    name: str
    dtype: str
    primary: bool = False


@dataclass
class NodeSchema:
    """Schema definition for a node table"""
    table_name: str
    properties: Dict[str, str]  # property_name: data_type
    primary_key: str

    def to_cypher(self) -> str:
        props = ", ".join(
            f"{k} {v}" for k, v in self.properties.items()
        )
        return (
            f"CREATE NODE TABLE {self.table_name}"
            f"({props}, PRIMARY KEY ({self.primary_key}))"
        )


@dataclass
class RelationshipSchema:
    """Schema definition for a relationship table"""
    table_name: str
    from_table: str
    to_table: str
    properties: Dict[str, str] = field(default_factory=dict)

    def to_cypher(self) -> str:
        props = ""
        if self.properties:
            props = ", " + ", ".join(
                f"{k} {v}" for k, v in self.properties.items()
            )
        return (
            f"CREATE REL TABLE {self.table_name}"
            f"(FROM {self.from_table} TO {self.to_table}{props})"
        )


@dataclass
class Node:
    """Represents a graph node"""
    table_name: str
    properties: Dict[str, Any]
    node_id: Optional[int] = None

    def __repr__(self):
        return f"Node(table={self.table_name}, props={self.properties})"


@dataclass
class Relationship:
    """Represents a graph relationship"""
    table_name: str
    from_node: Node
    to_node: Node
    properties: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"Relationship({self.from_node.table_name}"
            f"-[{self.table_name}]->{self.to_node.table_name})"
        )


@dataclass
class QueryResult:
    """Represents a query result"""
    columns: List[str]
    rows: List[Dict[str, Any]]
    execution_time: float
    row_count: int

    def to_list(self) -> List[Dict[str, Any]]:
        return self.rows

    def first(self) -> Optional[Dict[str, Any]]:
        return self.rows[0] if self.rows else None

    def is_empty(self) -> bool:
        return len(self.rows) == 0

    def __repr__(self):
        return (
            f"QueryResult(rows={self.row_count}, "
            f"time={self.execution_time:.4f}s)"
        )
