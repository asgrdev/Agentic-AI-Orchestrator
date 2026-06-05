# test_kuzu.py
"""
تست کامل کدهای جدید KuzuDB
سازگار با: KuzuClient, KuzuHelper, KuzuManager, AsyncKuzuManager
"""

from kuzudb_package import (
    KuzuClient,
    KuzuManager,
    KuzuHelper,
    GraphNode,
    GraphEdge,
    NodeSchema,
    RelationshipSchema,
    ColumnDef,
    AsyncKuzuManager,
)
import asyncio


# ─────────────────────────────────────────────
# تست Sync
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  KuzuDB New Package Test")
    print("=" * 50)

    # ── 1. Initialize ──────────────────────────────
    client = KuzuClient("./test_db")

    with client:
        helper  = KuzuHelper(client)
        manager = KuzuManager(client)

        # ── 2. Define Schemas ──────────────────────
        person_schema = NodeSchema(
            table_name="Person",
            properties=[
                ColumnDef("name",  "STRING", primary=True),
                ColumnDef("age",   "INT64"),
                ColumnDef("email", "STRING"),
                ColumnDef("city",  "STRING"),
            ],
            primary_key="name"
        )

        company_schema = NodeSchema(
            table_name="Company",
            properties=[
                ColumnDef("name",     "STRING", primary=True),
                ColumnDef("industry", "STRING"),
                ColumnDef("founded",  "INT64"),
            ],
            primary_key="name"

        )

        knows_schema = RelationshipSchema(
            table_name="KNOWS",
            from_table="Person",
            to_table="Person",
            properties=[
                ColumnDef("since",    "INT64"),
                ColumnDef("strength", "DOUBLE"),
            ],
            
        )

        works_at_schema = RelationshipSchema(
            table_name="WORKS_AT",
            from_table="Person",
            to_table="Company",
            properties=[
                ColumnDef("role",   "STRING"),
                ColumnDef("salary", "DOUBLE"),
            ],
        )

        # ── 3. Setup Schema ────────────────────────
        print("\n📐 Setting up Schema...")
        for schema in [person_schema, company_schema]:
            helper.create_node_table(schema)
        for schema in [knows_schema, works_at_schema]:
            helper.create_rel_table(schema)
        print("   Schema ready.")

        # ── 4. Insert Nodes ────────────────────────
        print("\n📌 Inserting People...")
        people = [
            {"name": "Alice", "age": 30, "email": "alice@example.com", "city": "Tehran"},
            {"name": "Bob",   "age": 25, "email": "bob@example.com",   "city": "Isfahan"},
            {"name": "Carol", "age": 35, "email": "carol@example.com", "city": "Shiraz"},
            {"name": "David", "age": 28, "email": "david@example.com", "city": "Tehran"},
        ]
        count = helper.insert_nodes_bulk("Person", people, batch_size=10)
        print(f"   {count} نفر درج شد.")

        print("\n🏢 Inserting Companies...")
        companies = [
            {"name": "TechCorp", "industry": "Technology",     "founded": 2010},
            {"name": "DataInc",  "industry": "Data Analytics", "founded": 2015},
        ]
        count = helper.insert_nodes_bulk("Company", companies, batch_size=10)
        print(f"   {count} شرکت درج شد.")

        # ── 5. Create Relationships ────────────────
        print("\n🔗 Creating Relationships...")

        helper.create_relationship(
            from_table="Person", from_props={"name": "Alice"},
            rel_table="KNOWS",
            to_table="Person",   to_props={"name": "Bob"},
            rel_props={"since": 2020, "strength": 0.9},
        )
        helper.create_relationship(
            from_table="Person", from_props={"name": "Bob"},
            rel_table="KNOWS",
            to_table="Person",   to_props={"name": "Carol"},
            rel_props={"since": 2021, "strength": 0.7},
        )
        helper.create_relationship(
            from_table="Person", from_props={"name": "Alice"},
            rel_table="WORKS_AT",
            to_table="Company",  to_props={"name": "TechCorp"},
            rel_props={"role": "Engineer", "salary": 90000.0},
        )
        helper.create_relationship(
            from_table="Person", from_props={"name": "Bob"},
            rel_table="WORKS_AT",
            to_table="Company",  to_props={"name": "DataInc"},
            rel_props={"role": "Analyst", "salary": 75000.0},
        )
        print("   روابط ساخته شدند.")

        # ── 6. Query Examples ──────────────────────
        print("\n🔍 Finding Alice...")
        result = helper.find_node("Person", {"name": "Alice"})
        print(f"   Found: {result.first()}")

        print("\n👥 All People:")
        all_people = helper.find_node("Person", {}, limit=50)
        for row in all_people:
            print(f"   - {row}")

        print("\n🤝 Alice's Neighbors (depth=2):")
        neighbors = helper.find_neighbors(
            table_name="Person",
            props={"name": "Alice"},
            depth=2,
            direction="out",
        )
        for row in neighbors:
            print(f"   - {row}")

        # ── 7. Raw Cypher Query ────────────────────
        print("\n🔥 Custom Cypher Query:")
        result = client.execute(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) "
            "RETURN p.name AS person, c.name AS company, "
            "c.industry AS industry"
        )
        for row in result:
            print(f"   {row['person']} → {row['company']} ({row['industry']})")

        # ── 8. Update Node ─────────────────────────
        print("\n✏️  Updating Alice's city...")
        helper.update_node(
            table_name="Person",
            match_props={"name": "Alice"},
            update_props={"city": "Mashhad"},
        )
        updated = helper.find_node("Person", {"name": "Alice"}).first()
        print(f"   Updated: {updated}")

        # ── 9. Transaction Example ─────────────────
        print("\n💼 Transaction Example...")
        try:
            with client.transaction():
                helper.insert_node("Person", {
                    "name": "Eve", "age": 22,
                    "email": "eve@example.com", "city": "Tehran",
                })
                helper.insert_node("Person", {
                    "name": "Frank", "age": 45,
                    "email": "frank@example.com", "city": "Tabriz",
                })
            print("   Transaction committed!")
        except Exception as e:
            print(f"   Transaction failed: {e}")

        # ── 10. KuzuManager - Entity & Relation ─
        print("\n🧠 KuzuManager - Knowledge Graph Test...")
        manager.setup_schema()

        node_a = GraphNode(
            id="ent_001", label="Person", name="علی",
            properties={"description": "مهندس نرم‌افزار", "source": "test"},
        )
        node_b = GraphNode(
            id="ent_002", label="Organization", name="شرکت X",
            properties={"description": "شرکت فناوری", "source": "test"},
        )
        edge = GraphEdge(
            source_id="ent_001",
            target_id="ent_002",
            relation_type="WORKS_AT",
            weight=1.0,
        )

        manager.upsert_entity(node_a)
        manager.upsert_entity(node_b)
        manager.upsert_relation(edge)
        print("   Entity و Relation ساخته شدند.")

        # بار دوم upsert - نباید duplicate بسازد
        manager.upsert_entity(node_a)
        print("   upsert دوباره: بدون duplicate.")

        ctx = manager.get_entity_context("ent_001")
        print(f"   Context علی: {ctx}")

        neighbors_kg = manager.get_neighbors("ent_001", hops=1)
        print(f"   همسایگان علی: {list(neighbors_kg)}")

        # ── 11. Document & Chunk ───────────────────
        print("\n📄 Document & Chunk Test...")
        manager.insert_document(
            doc_id="doc_001",
            title="مقاله تست",
            content="این یک مقاله آزمایشی است.",
            source_url="https://example.com/article1",
        )
        manager.insert_chunk(
            chunk_id="chunk_001",
            doc_id="doc_001",
            content="این یک مقاله آزمایشی است.",
            chunk_index=0,
        )
        manager.link_chunk_to_entity(
            chunk_id="chunk_001",
            entity_id="ent_001",
            confidence=0.95,
        )
        chunks = manager.get_chunks_by_entity("ent_001")
        print(f"   Chunk های مرتبط با علی: {list(chunks)}")

        # ── 12. Statistics ─────────────────────────
        print("\n📊 Statistics:")
        person_count  = helper.count_nodes("Person")
        company_count = helper.count_nodes("Company")
        kg_stats      = manager.stats()

        print(f"   Person  count : {person_count}")
        print(f"   Company count : {company_count}")
        print(f"   KG stats      : {kg_stats}")

        # ── 13. Shortest Path ──────────────────────
        print("\n🗺️  Shortest Path Test...")
        path = manager.find_path("ent_001", "ent_002")
        print(f"   Path result: {path.first()}")

        # ── 14. Health Check ───────────────────────
        healthy = manager.health_check()
        print(f"\n❤️  Health Check: {'OK' if healthy else 'FAILED'}")

    print("\n✅ Sync Test Complete!")


# ─────────────────────────────────────────────
# تست Async
# ─────────────────────────────────────────────

async def async_main():
    print("\n" + "=" * 50)
    print("  AsyncKuzuManager Test")
    print("=" * 50)

    async with AsyncKuzuManager("./test_db_async", max_workers=4) as mgr:

        # ── Entity ────────────────────────────────
        print("\n🧠 Async Entity Upsert...")
        node_x = GraphNode(
            id="async_001", label="Person", name="سارا",
            properties={"description": "متخصص داده", "source": "async_test"},
        )
        node_y = GraphNode(
            id="async_002", label="Organization", name="آزمایشگاه AI",
            properties={"description": "مرکز تحقیقاتی", "source": "async_test"},
        )
        edge_xy = GraphEdge(
            source_id="async_001",
            target_id="async_002",
            relation_type="MEMBER_OF",
            weight=1.5,
        )

        await mgr.upsert_entity(node_x)
        await mgr.upsert_entity(node_y)
        await mgr.upsert_relation(edge_xy)
        print("   Async entity و relation ساخته شدند.")

        # ── batch_upsert ──────────────────────────
        print("\n📦 Async Batch Upsert...")
        batch_nodes = [
            GraphNode(id=f"b_{i:03d}", label="Item", name=f"آیتم {i}")
            for i in range(5)
        ]
        batch_edges = [
            GraphEdge(
                source_id=f"b_{i:03d}",
                target_id=f"b_{i+1:03d}",
                relation_type="NEXT",
            )
            for i in range(4)
        ]
        result = await mgr.batch_upsert(batch_nodes, batch_edges)
        print(f"   Batch result: {result}")

        # ── Document & Chunk ──────────────────────
        print("\n📄 Async Document & Chunk...")
        await mgr.insert_document(
            doc_id="async_doc_001",
            title="سند async",
            content="محتوای سند غیرهمزمان.",
            source_url="https://example.com/async",
        )
        await mgr.insert_chunk(
            chunk_id="async_chunk_001",
            doc_id="async_doc_001",
            content="محتوای سند غیرهمزمان.",
            chunk_index=0,
        )
        await mgr.link_chunk_to_entity(
            chunk_id="async_chunk_001",
            entity_id="async_001",
            confidence=0.88,
        )

        # ── Query ──────────────────────────────────
        print("\n🔍 Async Find Entity...")
        found = await mgr.find_entity_by_name("سارا")
        print(f"   Found: {found.first()}")

        ctx = await mgr.get_entity_context("async_001")
        print(f"   Context: {ctx}")

        neighbors = await mgr.get_neighbors("async_001", hops=1)
        print(f"   Neighbors: {list(neighbors)}")

        chunks = await mgr.get_chunks_by_entity("async_001")
        print(f"   Chunks: {list(chunks)}")

        path = await mgr.find_path("async_001", "async_002")
        print(f"   Path: {path.first()}")

        # ── Stats & Health ─────────────────────────
        stats   = await mgr.stats()
        healthy = await mgr.health_check()
        print(f"\n📊 Async Stats   : {stats}")
        print(f"❤️  Health Check : {'OK' if healthy else 'FAILED'}")

    print("\n✅ Async Test Complete!")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()
    asyncio.run(async_main())
