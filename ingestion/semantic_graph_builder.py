"""
Semantic Graph Builder - ساخت گراف معنایی پیشرفته

این ماژول گراف دانش را با روابط معنایی غنی‌تر می‌سازد:
- استخراج روابط معنایی از متن
- ایجاد روابط استنتاجی (inferred relations)
- محاسبه وزن و اهمیت روابط
- ادغام هوشمند دانش جدید با دانش موجود
- نسخه‌بندی و تاریخچه تغییرات
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
from enum import Enum

if TYPE_CHECKING:
    from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager
    from ingestion.datasets.entities import Entity
    from ingestion.datasets.relations import Relation

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """انواع روابط در گراف معنایی"""
    EXPLICIT = "explicit"          # روابط صریح از متن
    INFERRED = "inferred"          # روابط استنتاجی
    SEMANTIC = "semantic"          # روابط معنایی (مترادف، ضد، ...)
    TEMPORAL = "temporal"          # روابط زمانی
    CAUSAL = "causal"             # روابط علت و معلولی
    HIERARCHICAL = "hierarchical"  # روابط سلسله‌مراتبی


@dataclass
class SemanticRelation:
    """رابطه معنایی با متادیتای کامل"""
    subject_id: str
    predicate: str
    object_id: str
    relation_type: RelationType
    confidence: float
    source_chunk: str
    source_doc: str
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphUpdate:
    """نتیجه بروزرسانی گراف"""
    nodes_added: int = 0
    nodes_updated: int = 0
    edges_added: int = 0
    edges_updated: int = 0
    inferred_relations: int = 0
    conflicts_resolved: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class SemanticGraphBuilder:
    """
    سازنده گراف معنایی با قابلیت‌های پیشرفته:
    - ادغام هوشمند دانش
    - استنتاج روابط جدید
    - حل تعارض
    - نسخه‌بندی
    """

    def __init__(
        self,
        graph_manager: "AsyncKuzuManager",
        enable_inference: bool = True,
        enable_versioning: bool = True,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.graph = graph_manager
        self.enable_inference = enable_inference
        self.enable_versioning = enable_versioning
        self.confidence_threshold = confidence_threshold
        
        # کش برای جلوگیری از query های تکراری
        self._entity_cache: dict[str, dict] = {}
        self._relation_cache: dict[str, list[SemanticRelation]] = {}

    # ══════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════

    async def build_semantic_graph(
        self,
        chunk: dict,
        entities: list["Entity"],
        relations: list["Relation"],
    ) -> GraphUpdate:
        """
        ساخت گراف معنایی کامل از chunk + entities + relations
        
        Args:
            chunk: دیکشنری شامل id, title, text
            entities: لیست entities استخراج شده
            relations: لیست relations استخراج شده
            
        Returns:
            GraphUpdate با آمار تغییرات
        """
        update = GraphUpdate()
        
        # 1. پردازش و ادغام entities
        processed_entities = await self._process_entities(entities, chunk)
        update.nodes_added = len([e for e in processed_entities if e.get("is_new")])
        update.nodes_updated = len([e for e in processed_entities if not e.get("is_new")])
        
        # 2. پردازش و ادغام relations
        semantic_relations = await self._process_relations(
            relations, entities, chunk
        )
        update.edges_added = len([r for r in semantic_relations if r.version == 1])
        update.edges_updated = len([r for r in semantic_relations if r.version > 1])
        
        # 3. استنتاج روابط جدید
        if self.enable_inference:
            inferred = await self._infer_relations(processed_entities, semantic_relations)
            semantic_relations.extend(inferred)
            update.inferred_relations = len(inferred)
        
        # 4. درج در گراف
        await self._upsert_to_graph(processed_entities, semantic_relations, chunk)
        
        # 5. ایجاد روابط معنایی (synonyms, hierarchies, etc.)
        await self._build_semantic_links(processed_entities)
        
        logger.info(
            "Semantic graph built: +%d nodes, ~%d nodes, +%d edges, ~%d edges, %d inferred",
            update.nodes_added, update.nodes_updated,
            update.edges_added, update.edges_updated,
            update.inferred_relations
        )
        
        return update

    async def incremental_update(
        self,
        new_entities: list["Entity"],
        new_relations: list["Relation"],
        chunk: dict,
    ) -> GraphUpdate:
        """
        بروزرسانی تدریجی گراف با دانش جدید
        - حل تعارض با دانش موجود
        - نگهداری تاریخچه تغییرات
        """
        update = GraphUpdate()
        
        # بررسی entities موجود
        for entity in new_entities:
            existing = await self._find_existing_entity(entity)
            
            if existing:
                # ادغام با entity موجود
                merged = await self._merge_entity(existing, entity, chunk)
                if merged.get("updated"):
                    update.nodes_updated += 1
            else:
                # entity جدید
                await self._create_entity(entity, chunk)
                update.nodes_added += 1
        
        # بررسی relations موجود
        for relation in new_relations:
            existing = await self._find_existing_relation(relation)
            
            if existing:
                # حل تعارض
                resolved = await self._resolve_conflict(existing, relation, chunk)
                if resolved:
                    update.conflicts_resolved += 1
                    update.edges_updated += 1
            else:
                # relation جدید
                await self._create_relation(relation, chunk)
                update.edges_added += 1
        
        return update

    # ══════════════════════════════════════════════════════════════
    # Entity Processing
    # ══════════════════════════════════════════════════════════════

    async def _process_entities(
        self,
        entities: list["Entity"],
        chunk: dict,
    ) -> list[dict]:
        """پردازش و غنی‌سازی entities"""
        processed = []
        
        for entity in entities:
            # بررسی وجود در گراف
            existing = await self._find_existing_entity(entity)
            
            entity_dict = {
                "id": entity.linked_id or entity.id,
                "name": entity.text,
                "canonical_name": entity.canonical or entity.text,
                "type": entity.label or entity.type,
                "confidence": entity.confidence,
                "source_chunk": chunk["id"],
                "source_doc": chunk.get("title", ""),
                "is_new": existing is None,
            }
            
            if existing:
                # ادغام با entity موجود
                entity_dict["version"] = existing.get("version", 1) + 1
                entity_dict["previous_names"] = existing.get("previous_names", [])
                if entity.text not in entity_dict["previous_names"]:
                    entity_dict["previous_names"].append(entity.text)
            else:
                entity_dict["version"] = 1
                entity_dict["previous_names"] = []
            
            processed.append(entity_dict)
        
        return processed

    async def _find_existing_entity(self, entity: "Entity") -> Optional[dict]:
        """جستجوی entity در گراف"""
        entity_id = entity.linked_id or entity.id
        
        # بررسی کش
        if entity_id in self._entity_cache:
            return self._entity_cache[entity_id]
        
        try:
            result = await self.graph.execute(
                """
                MATCH (e:Entity {id: $id})
                RETURN e.id AS id, e.name AS name, e.type AS type,
                       e.version AS version, e.previous_names AS previous_names
                """,
                parameters={"id": entity_id}
            )
            
            if result.rows:
                entity_dict = dict(result.rows[0])
                self._entity_cache[entity_id] = entity_dict
                return entity_dict
        except Exception as e:
            logger.warning("Entity lookup failed: %s", e)
        
        return None

    async def _merge_entity(
        self,
        existing: dict,
        new_entity: "Entity",
        chunk: dict,
    ) -> dict:
        """ادغام entity جدید با موجود"""
        # محاسبه confidence جدید (میانگین وزن‌دار)
        old_conf = existing.get("confidence", 0.5)
        new_conf = new_entity.confidence
        merged_conf = (old_conf * 0.7 + new_conf * 0.3)  # وزن بیشتر به دانش قدیمی
        
        # بروزرسانی
        try:
            await self.graph.execute(
                """
                MATCH (e:Entity {id: $id})
                SET e.confidence = $confidence,
                    e.version = e.version + 1,
                    e.last_updated = $timestamp,
                    e.last_seen_chunk = $chunk_id
                """,
                parameters={
                    "id": existing["id"],
                    "confidence": merged_conf,
                    "timestamp": datetime.now().isoformat(),
                    "chunk_id": chunk["id"],
                }
            )
            return {"updated": True, "confidence": merged_conf}
        except Exception as e:
            logger.error("Entity merge failed: %s", e)
            return {"updated": False}

    async def _create_entity(self, entity: "Entity", chunk: dict) -> None:
        """ایجاد entity جدید"""
        from knowledg_graph.kuzudb_package.models import GraphNode
        
        node = GraphNode(
            id=entity.linked_id or entity.id,
            name=entity.text,
            canonical_name=entity.canonical or entity.text,
            type=entity.label or entity.type,
            confidence=entity.confidence,
            linked_id=entity.linked_id or "",
            description=getattr(entity, "description", ""),
            kb_url=getattr(entity, "kb_url", ""),
        )
        
        await self.graph.upsert_entity(node)

    # ══════════════════════════════════════════════════════════════
    # Relation Processing
    # ══════════════════════════════════════════════════════════════

    async def _process_relations(
        self,
        relations: list["Relation"],
        entities: list["Entity"],
        chunk: dict,
    ) -> list[SemanticRelation]:
        """پردازش و تبدیل relations به SemanticRelation"""
        semantic_relations = []
        
        for relation in relations:
            # فیلتر روابط با confidence پایین
            if relation.confidence < self.confidence_threshold:
                continue
            
            sem_rel = SemanticRelation(
                subject_id=relation.subject_id or relation.subject,
                predicate=relation.predicate,
                object_id=relation.object_id or relation.object,
                relation_type=RelationType.EXPLICIT,
                confidence=relation.confidence,
                source_chunk=chunk["id"],
                source_doc=chunk.get("title", ""),
                evidence=[chunk.get("text", "")[:200]],
            )
            
            # بررسی وجود relation مشابه
            existing = await self._find_existing_relation(relation)
            if existing:
                sem_rel.version = existing.version + 1
                sem_rel.evidence.extend(existing.evidence[:3])  # حداکثر 3 شاهد قبلی
            
            semantic_relations.append(sem_rel)
        
        return semantic_relations

    async def _find_existing_relation(self, relation: "Relation") -> Optional[SemanticRelation]:
        """جستجوی relation در گراف"""
        cache_key = f"{relation.subject_id}:{relation.predicate}:{relation.object_id}"
        
        if cache_key in self._relation_cache:
            return self._relation_cache[cache_key][0] if self._relation_cache[cache_key] else None
        
        try:
            result = await self.graph.execute(
                """
                MATCH (a:Entity {id: $subject})-[r:RELATION {rel_type: $predicate}]->(b:Entity {id: $object})
                RETURN r.version AS version, r.confidence AS confidence,
                       r.evidence AS evidence, r.timestamp AS timestamp
                LIMIT 1
                """,
                parameters={
                    "subject": relation.subject_id or relation.subject,
                    "predicate": relation.predicate,
                    "object": relation.object_id or relation.object,
                }
            )
            
            if result.rows:
                row = result.rows[0]
                sem_rel = SemanticRelation(
                    subject_id=relation.subject_id or relation.subject,
                    predicate=relation.predicate,
                    object_id=relation.object_id or relation.object,
                    relation_type=RelationType.EXPLICIT,
                    confidence=row.get("confidence", 0.5),
                    source_chunk="",
                    source_doc="",
                    version=row.get("version", 1),
                    evidence=row.get("evidence", []),
                )
                return sem_rel
        except Exception as e:
            logger.warning("Relation lookup failed: %s", e)
        
        return None

    async def _create_relation(self, relation: "Relation", chunk: dict) -> None:
        """ایجاد relation جدید"""
        from knowledg_graph.kuzudb_package.models import GraphEdge
        
        edge = GraphEdge(
            from_id=relation.subject_id or relation.subject,
            to_id=relation.object_id or relation.object,
            rel_type=relation.predicate,
            weight=relation.confidence,
            confidence=relation.confidence,
            source_doc=chunk.get("title", ""),
            source_chunk=chunk["id"],
        )
        
        await self.graph.upsert_relation(edge)

    async def _resolve_conflict(
        self,
        existing: SemanticRelation,
        new_relation: "Relation",
        chunk: dict,
    ) -> bool:
        """
        حل تعارض بین relation موجود و جدید
        استراتژی: میانگین وزن‌دار confidence + نگهداری شواهد
        """
        # محاسبه confidence جدید
        old_conf = existing.confidence
        new_conf = new_relation.confidence
        
        # اگر تفاوت زیاد باشد، وزن بیشتری به جدید می‌دهیم
        if abs(new_conf - old_conf) > 0.3:
            merged_conf = (old_conf * 0.4 + new_conf * 0.6)
        else:
            merged_conf = (old_conf * 0.6 + new_conf * 0.4)
        
        try:
            await self.graph.execute(
                """
                MATCH (a:Entity {id: $subject})-[r:RELATION {rel_type: $predicate}]->(b:Entity {id: $object})
                SET r.confidence = $confidence,
                    r.version = r.version + 1,
                    r.last_updated = $timestamp
                """,
                parameters={
                    "subject": existing.subject_id,
                    "predicate": existing.predicate,
                    "object": existing.object_id,
                    "confidence": merged_conf,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return True
        except Exception as e:
            logger.error("Conflict resolution failed: %s", e)
            return False

    # ══════════════════════════════════════════════════════════════
    # Relation Inference
    # ══════════════════════════════════════════════════════════════

    async def _infer_relations(
        self,
        entities: list[dict],
        relations: list[SemanticRelation],
    ) -> list[SemanticRelation]:
        """
        استنتاج روابط جدید از روابط موجود
        مثال: A→B و B→C ⇒ A→C (transitive)
        """
        inferred = []
        
        # قوانین استنتاج
        inferred.extend(await self._infer_transitive(relations))
        inferred.extend(await self._infer_symmetric(relations))
        inferred.extend(await self._infer_hierarchical(entities))
        
        return inferred

    async def _infer_transitive(
        self,
        relations: list[SemanticRelation],
    ) -> list[SemanticRelation]:
        """استنتاج روابط متعدی (transitive)"""
        inferred = []
        transitive_predicates = {"part_of", "located_in", "member_of", "subclass_of"}
        
        for r1 in relations:
            if r1.predicate not in transitive_predicates:
                continue
            
            for r2 in relations:
                if r2.predicate != r1.predicate:
                    continue
                
                # A→B و B→C ⇒ A→C
                if r1.object_id == r2.subject_id:
                    # بررسی عدم وجود رابطه مستقیم
                    direct_exists = any(
                        r.subject_id == r1.subject_id and r.object_id == r2.object_id
                        for r in relations
                    )
                    
                    if not direct_exists:
                        inferred_rel = SemanticRelation(
                            subject_id=r1.subject_id,
                            predicate=r1.predicate,
                            object_id=r2.object_id,
                            relation_type=RelationType.INFERRED,
                            confidence=min(r1.confidence, r2.confidence) * 0.8,  # کاهش confidence
                            source_chunk="inferred",
                            source_doc="inference_engine",
                            evidence=[f"Inferred from {r1.subject_id}→{r1.object_id}→{r2.object_id}"],
                        )
                        inferred.append(inferred_rel)
        
        return inferred

    async def _infer_symmetric(
        self,
        relations: list[SemanticRelation],
    ) -> list[SemanticRelation]:
        """استنتاج روابط متقارن (symmetric)"""
        inferred = []
        symmetric_predicates = {"similar_to", "related_to", "colleague_of", "sibling_of"}
        
        for relation in relations:
            if relation.predicate not in symmetric_predicates:
                continue
            
            # A→B ⇒ B→A
            reverse_exists = any(
                r.subject_id == relation.object_id and r.object_id == relation.subject_id
                for r in relations
            )
            
            if not reverse_exists:
                inferred_rel = SemanticRelation(
                    subject_id=relation.object_id,
                    predicate=relation.predicate,
                    object_id=relation.subject_id,
                    relation_type=RelationType.INFERRED,
                    confidence=relation.confidence * 0.9,
                    source_chunk="inferred",
                    source_doc="inference_engine",
                    evidence=[f"Symmetric of {relation.subject_id}→{relation.object_id}"],
                )
                inferred.append(inferred_rel)
        
        return inferred

    async def _infer_hierarchical(self, entities: list[dict]) -> list[SemanticRelation]:
        """استنتاج روابط سلسله‌مراتبی از انواع entities"""
        inferred = []
        
        # گروه‌بندی entities بر اساس type
        type_groups: dict[str, list[dict]] = {}
        for entity in entities:
            etype = entity.get("type", "")
            if etype:
                type_groups.setdefault(etype, []).append(entity)
        
        # ایجاد روابط instance_of
        for etype, group_entities in type_groups.items():
            for entity in group_entities:
                inferred_rel = SemanticRelation(
                    subject_id=entity["id"],
                    predicate="instance_of",
                    object_id=f"TYPE:{etype}",
                    relation_type=RelationType.HIERARCHICAL,
                    confidence=0.95,
                    source_chunk="inferred",
                    source_doc="type_inference",
                    evidence=[f"{entity['name']} is of type {etype}"],
                )
                inferred.append(inferred_rel)
        
        return inferred

    # ══════════════════════════════════════════════════════════════
    # Semantic Links
    # ══════════════════════════════════════════════════════════════

    async def _build_semantic_links(self, entities: list[dict]) -> None:
        """ایجاد روابط معنایی بین entities (مترادف، ضد، ...)"""
        # این بخش می‌تواند با استفاده از WordNet یا مدل‌های زبانی پیاده شود
        # فعلاً placeholder است
        pass

    # ══════════════════════════════════════════════════════════════
    # Graph Operations
    # ══════════════════════════════════════════════════════════════

    async def _upsert_to_graph(
        self,
        entities: list[dict],
        relations: list[SemanticRelation],
        chunk: dict,
    ) -> None:
        """درج entities و relations در گراف"""
        from knowledg_graph.kuzudb_package.models import GraphNode, GraphEdge
        
        # تبدیل به GraphNode
        nodes = [
            GraphNode(
                id=e["id"],
                name=e["name"],
                canonical_name=e["canonical_name"],
                type=e["type"],
                confidence=e["confidence"],
                linked_id="",
                description="",
                kb_url="",
            )
            for e in entities
        ]
        
        # تبدیل به GraphEdge
        edges = [
            GraphEdge(
                from_id=r.subject_id,
                to_id=r.object_id,
                rel_type=r.predicate,
                weight=r.confidence,
                confidence=r.confidence,
                source_doc=r.source_doc,
                source_chunk=r.source_chunk,
            )
            for r in relations
        ]
        
        # Batch upsert
        await self.graph.batch_upsert(nodes, edges)
        
        # ایجاد MENTIONS edges
        for entity in entities:
            try:
                await self.graph.execute(
                    """
                    MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $entity_id})
                    MERGE (c)-[m:MENTIONS]->(e)
                    SET m.confidence = $confidence
                    """,
                    parameters={
                        "chunk_id": chunk["id"],
                        "entity_id": entity["id"],
                        "confidence": entity["confidence"],
                    }
                )
            except Exception as e:
                logger.warning("MENTIONS edge creation failed: %s", e)

    # ══════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════

    def clear_cache(self) -> None:
        """پاک کردن کش"""
        self._entity_cache.clear()
        self._relation_cache.clear()

    async def get_graph_stats(self) -> dict[str, Any]:
        """آمار گراف"""
        stats = {"entity_count": 0, "relation_count": 0}
        try:
            # دو کوئری جدا — کوئری ترکیبی وقتی RELATION خالی است هیچ سطری برنمی‌گرداند
            ent = await self.graph.execute(
                "MATCH (e:Entity) RETURN count(e) AS entity_count"
            )
            if ent.rows:
                stats["entity_count"] = ent.rows[0].get("entity_count", 0)

            rel = await self.graph.execute(
                "MATCH ()-[r:RELATION]->() RETURN count(r) AS relation_count"
            )
            if rel.rows:
                stats["relation_count"] = rel.rows[0].get("relation_count", 0)
        except Exception as e:
            logger.error("Stats query failed: %s", e)

        return stats

