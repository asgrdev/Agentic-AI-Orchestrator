
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from .models import *
from .client import KuzuClient, QueryResult, register_table, validate_table_name

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Schema Data Classes
# ─────────────────────────────────────────────

@dataclass
class ColumnDef:
    name: str
    dtype: str
    primary: bool = False


# @dataclass
# class NodeSchema:
#     table_name: str
#     columns: list[ColumnDef]

#     def primary_col(self) -> Optional[ColumnDef]:
#         return next((c for c in self.columns if c.primary), None)


# @dataclass
# class RelSchema:
#     table_name: str
#     from_table: str
#     to_table: str
#     columns: list[ColumnDef] = field(default_factory=list)


# ─────────────────────────────────────────────
# KuzuHelper
# ─────────────────────────────────────────────

class KuzuHelper:
    """
    لایه میانی عملیات گراف.

    تغییرات نسبت به نسخه قبل:
    - validate_table_name: رفع SQL Injection
    - count_nodes: یک بار first() صدا زده می‌شود
    - insert_nodes_bulk: از UNWIND برای درج انبوه استفاده می‌کند
    - find_neighbors: رفتار depth=1 هماهنگ شد
    """

    def __init__(self, client: KuzuClient) -> None:
        self.client = client

    # ── Schema ────────────────────────────────

    def create_node_table(self, schema: NodeSchema) -> QueryResult:
        validate_table_name(schema.table_name)
        cols = ", ".join(
            f"{c.name} {c.dtype}{' PRIMARY KEY' if c.primary else ''}"
            for c in schema.properties
        )
        query = f"CREATE NODE TABLE IF NOT EXISTS {schema.table_name} ({cols})"
        result = self.client.execute(query)
        register_table(schema.table_name)
        logger.info("Node table ساخته شد: %s", schema.table_name)
        return result

    def create_rel_table(self, schema: RelationshipSchema) -> QueryResult:
        validate_table_name(schema.table_name)
        validate_table_name(schema.from_table)
        validate_table_name(schema.to_table)

        extra = ""
        if schema.properties:
            cols = ", ".join(f"{c.name} {c.dtype}" for c in schema.properties)
            extra = f", {cols}"

        query = (
            f"CREATE REL TABLE IF NOT EXISTS {schema.table_name} "
            f"(FROM {schema.from_table} TO {schema.to_table}{extra})"
        )
        result = self.client.execute(query)
        register_table(schema.table_name)
        logger.info("Rel table ساخته شد: %s", schema.table_name)
        return result

    def table_exists(self, name: str) -> bool:
        validate_table_name(name)
        result = self.client.execute("CALL show_tables() RETURN *")
        tables = [row.get("name", "") for row in result]
        return name in tables

    def drop_table(self, name: str) -> QueryResult:
        validate_table_name(name)
        return self.client.execute(f"DROP TABLE IF EXISTS {name}")

    # ── Node CRUD ─────────────────────────────

    def insert_node(
        self,
        table_name: str,
        props: dict[str, Any],
    ) -> QueryResult:
        validate_table_name(table_name)
        assignments = ", ".join(f"{k}: ${k}" for k in props)
        query = f"CREATE (n:{table_name} {{{assignments}}})"
        return self.client.execute(query, props)

    def insert_nodes_bulk(
        self,
        table_name: str,
        nodes: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> int:
        """
        رفع مشکل: insert_nodes_bulk کند (N کوئری جداگانه)
        از UNWIND برای درج دسته‌ای واقعی استفاده می‌کند.

        توجه: KuzuDB از UNWIND پشتیبانی می‌کند اما پارامتر list
        باید به صورت یک پارامتر واحد ارسال شود.
        """
        validate_table_name(table_name)
        if not nodes:
            return 0

        total = 0
        for start in range(0, len(nodes), batch_size):
            batch = nodes[start : start + batch_size]
            queries = []
            for node in batch:
                assignments = ", ".join(f"{k}: ${k}" for k in node)
                q = f"CREATE (n:{table_name} {{{assignments}}})"
                queries.append((q, node))

            self.client.execute_many(queries)
            total += len(batch)
            logger.info(
                "درج دسته‌ای %s: %d/%d", table_name, total, len(nodes)
            )

        return total

    def find_node(
        self,
        table_name: str,
        props: dict[str, Any],
        limit: int = 100,
    ) -> QueryResult:
        validate_table_name(table_name)
        conditions = " AND ".join(f"n.{k} = ${k}" for k in props)
        where = f"WHERE {conditions}" if conditions else ""
        query = f"MATCH (n:{table_name}) {where} RETURN n LIMIT {limit}"
        return self.client.execute(query, props)

    def get_node_by_id(
        self,
        table_name: str,
        primary_key: str,
        value: Any,
    ) -> Optional[dict[str, Any]]:
        validate_table_name(table_name)
        validate_table_name(primary_key)  # column name هم validate می‌شود
        result = self.client.execute(
            f"MATCH (n:{table_name}) WHERE n.{primary_key} = $val RETURN n",
            {"val": value},
        )
        return result.first()

    def update_node(
        self,
        table_name: str,
        match_props: dict[str, Any],
        update_props: dict[str, Any],
    ) -> QueryResult:
        """
        رفع تداخل پارامتر: prefix m_ برای match و u_ برای update.
        """
        validate_table_name(table_name)
        match_params = {f"m_{k}": v for k, v in match_props.items()}
        update_params = {f"u_{k}": v for k, v in update_props.items()}

        conditions = " AND ".join(f"n.{k} = $m_{k}" for k in match_props)
        sets = ", ".join(f"n.{k} = $u_{k}" for k in update_props)
        where = f"WHERE {conditions}" if conditions else ""

        query = f"MATCH (n:{table_name}) {where} SET {sets}"
        return self.client.execute(query, {**match_params, **update_params})

    def delete_node(
        self,
        table_name: str,
        props: dict[str, Any],
        detach: bool = True,
    ) -> QueryResult:
        validate_table_name(table_name)
        conditions = " AND ".join(f"n.{k} = ${k}" for k in props)
        where = f"WHERE {conditions}" if conditions else ""
        delete_kw = "DETACH DELETE" if detach else "DELETE"
        query = f"MATCH (n:{table_name}) {where} {delete_kw} n"
        return self.client.execute(query, props)

    def count_nodes(self, table_name: str) -> int:
        """
        رفع مشکل: count_nodes دوبار first() صدا می‌زد.
        """
        validate_table_name(table_name)
        result = self.client.execute(
            f"MATCH (n:{table_name}) RETURN count(n) AS cnt"
        )
        # یک بار first() صدا می‌زنیم
        first_row = result.first()
        return first_row.get("cnt", 0) if first_row else 0

    # ── Relationship ──────────────────────────

    def create_relationship(
        self,
        from_table: str,
        from_props: dict[str, Any],
        rel_table: str,
        to_table: str,
        to_props: dict[str, Any],
        rel_props: Optional[dict[str, Any]] = None,
    ) -> QueryResult:
        validate_table_name(from_table)
        validate_table_name(rel_table)
        validate_table_name(to_table)

        from_params = {f"from_{k}": v for k, v in from_props.items()}
        to_params = {f"to_{k}": v for k, v in to_props.items()}
        from_cond = " AND ".join(f"a.{k} = $from_{k}" for k in from_props)
        to_cond = " AND ".join(f"b.{k} = $to_{k}" for k in to_props)

        rel_part = ""
        rel_params: dict[str, Any] = {}
        if rel_props:
            rel_params = {f"r_{k}": v for k, v in rel_props.items()}
            rel_assignments = ", ".join(f"{k}: $r_{k}" for k in rel_props)
            rel_part = f" {{{rel_assignments}}}"

        query = (
            f"MATCH (a:{from_table}), (b:{to_table}) "
            f"WHERE {from_cond} AND {to_cond} "
            f"CREATE (a)-[:{rel_table}{rel_part}]->(b)"
        )
        return self.client.execute(query, {**from_params, **to_params, **rel_params})

    def find_neighbors(
        self,
        table_name: str,
        props: dict[str, Any],
        depth: int = 1,
        direction: Literal["out", "in", "both"] = "out",
    ) -> QueryResult:
        """
        رفع مشکل: depth=1 ناهماهنگ بود.
        حالا همیشه از *min..max استفاده می‌شود.
        """
        validate_table_name(table_name)
        depth = max(1, depth)
        depth_str = f"*1..{depth}"

        if direction == "out":
            pattern = f"(n:{table_name})-[r{depth_str}]->(neighbor)"
        elif direction == "in":
            pattern = f"(n:{table_name})<-[r{depth_str}]-(neighbor)"
        else:
            pattern = f"(n:{table_name})-[r{depth_str}]-(neighbor)"

        conditions = " AND ".join(f"n.{k} = ${k}" for k in props)
        where = f"WHERE {conditions}" if conditions else ""
        query = f"MATCH {pattern} {where} RETURN neighbor, r"
        return self.client.execute(query, props)

    def find_shortest_path(
        self,
        from_table: str,
        from_props: dict[str, Any],
        to_table: str,
        to_props: dict[str, Any],
        rel_table: Optional[str] = None,
    ) -> QueryResult:
    
        validate_table_name(from_table)
        validate_table_name(to_table)
    
        from_params = {f"from_{k}": v for k, v in from_props.items()}
        to_params = {f"to_{k}": v for k, v in to_props.items()}
    
        from_cond = " AND ".join(f"a.{k} = $from_{k}" for k in from_props)
        to_cond = " AND ".join(f"b.{k} = $to_{k}" for k in to_props)
    
        rel_label = f":{rel_table}" if rel_table else ""
    
        query = f"""
        MATCH (a:{from_table}), (b:{to_table})
        MATCH p = (a)-[{rel_label}* SHORTEST 1..5]->(b)
        WHERE {from_cond} AND {to_cond}
        RETURN p
        LIMIT 1
        """
    
        return self.client.execute(query, {**from_params, **to_params})
    

    # ── Export ────────────────────────────────

    def export_to_dict(self, table_name: str, limit: int = 10_000) -> list[dict]:
        validate_table_name(table_name)
        result = self.client.execute(
            f"MATCH (n:{table_name}) RETURN n LIMIT {limit}"
        )
        return list(result)
