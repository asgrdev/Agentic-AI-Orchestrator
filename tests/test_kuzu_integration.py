"""
Test KuzuDB Integration
========================

تست کامل یکپارچگی KuzuDB با سیستم.
"""

import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledg_graph.kuzu_wrapper import (
    KuzuAvailable,
    ensure_kuzu_available,
    get_kuzu_client_or_mock,
    print_kuzu_status,
    get_kuzu_info,
)


def test_kuzu_availability():
    """Test 1: بررسی در دسترس بودن Kuzu"""
    print("\n" + "="*60)
    print("Test 1: Kuzu Availability")
    print("="*60)
    
    print_kuzu_status()
    
    info = get_kuzu_info()
    print(f"\nKuzu Info: {info}")
    
    if KuzuAvailable:
        print("✅ Test 1 PASSED: Kuzu is available")
        return True
    else:
        print("⚠️  Test 1 WARNING: Kuzu not available (will use mock)")
        return False


def test_direct_kuzu_connection():
    """Test 2: اتصال مستقیم به Kuzu"""
    print("\n" + "="*60)
    print("Test 2: Direct Kuzu Connection")
    print("="*60)
    
    if not KuzuAvailable:
        print("⏭️  Test 2 SKIPPED: Kuzu not available")
        return True
    
    try:
        import kuzu
        
        # Clean up
        test_db = Path("./test_kuzu_direct")
        if test_db.exists():
            shutil.rmtree(test_db)
        
        # Create database
        print("Creating database...")
        db = kuzu.Database(str(test_db))
        conn = kuzu.Connection(db)
        print("✅ Database created")
        
        # Create schema
        print("Creating schema...")
        conn.execute("""
            CREATE NODE TABLE Entity(
                id STRING,
                name STRING,
                type STRING,
                PRIMARY KEY (id)
            )
        """)
        print("✅ Schema created")
        
        # Insert data
        print("Inserting data...")
        conn.execute("CREATE (:Entity {id: 'e1', name: 'Test Entity', type: 'TEST'})")
        print("✅ Data inserted")
        
        # Query data
        print("Querying data...")
        result = conn.execute("MATCH (e:Entity) RETURN e.id, e.name, e.type")
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        
        print(f"✅ Query returned {len(rows)} rows")
        for row in rows:
            print(f"   - {row}")
        
        # Clean up
        shutil.rmtree(test_db)
        
        print("✅ Test 2 PASSED: Direct connection works")
        return True
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        return False


def test_wrapper_connection():
    """Test 3: اتصال از طریق wrapper"""
    print("\n" + "="*60)
    print("Test 3: Wrapper Connection")
    print("="*60)
    
    test_db = Path("./test_kuzu_wrapper")
    if test_db.exists():
        shutil.rmtree(test_db)
    
    try:
        # Get client (real or mock)
        print("Getting client...")
        client = get_kuzu_client_or_mock(str(test_db), allow_mock=True)
        print(f"✅ Client obtained: {type(client).__name__}")
        
        if KuzuAvailable:
            # Test with real Kuzu
            print("Testing with real Kuzu...")
            
            # Create schema
            client.execute("""
                CREATE NODE TABLE TestNode(
                    id STRING,
                    value INT64,
                    PRIMARY KEY (id)
                )
            """)
            print("✅ Schema created")
            
            # Insert
            client.execute("CREATE (:TestNode {id: 't1', value: 42})")
            print("✅ Data inserted")
            
            # Query
            result = client.execute("MATCH (n:TestNode) RETURN n.id, n.value")
            print(f"✅ Query executed: {result}")
            
            # Clean up
            if hasattr(client, 'disconnect'):
                client.disconnect()
            shutil.rmtree(test_db)
        else:
            # Test with mock
            print("Testing with mock...")
            result = client.execute("MATCH (n) RETURN n")
            print(f"✅ Mock query executed: {result}")
        
        print("✅ Test 3 PASSED: Wrapper works")
        return True
        
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_graph_operations():
    """Test 4: عملیات گراف دانش"""
    print("\n" + "="*60)
    print("Test 4: Knowledge Graph Operations")
    print("="*60)
    
    if not KuzuAvailable:
        print("⏭️  Test 4 SKIPPED: Kuzu not available")
        return True
    
    test_db = Path("./test_kuzu_kg")
    if test_db.exists():
        shutil.rmtree(test_db)
    
    try:
        import kuzu
        
        # Create database
        db = kuzu.Database(str(test_db))
        conn = kuzu.Connection(db)
        
        # Create knowledge graph schema
        print("Creating KG schema...")
        conn.execute("""
            CREATE NODE TABLE Entity(
                id STRING,
                name STRING,
                canonical_name STRING,
                type STRING,
                description STRING,
                confidence DOUBLE,
                PRIMARY KEY (id)
            )
        """)
        
        conn.execute("""
            CREATE REL TABLE RELATION(
                FROM Entity TO Entity,
                type STRING,
                weight DOUBLE,
                confidence DOUBLE
            )
        """)
        print("✅ KG schema created")
        
        # Insert entities
        print("Inserting entities...")
        conn.execute("""
            CREATE (:Entity {
                id: 'e1',
                name: 'Python',
                canonical_name: 'Python Programming Language',
                type: 'TECHNOLOGY',
                description: 'A high-level programming language',
                confidence: 0.95
            })
        """)
        
        conn.execute("""
            CREATE (:Entity {
                id: 'e2',
                name: 'Machine Learning',
                canonical_name: 'Machine Learning',
                type: 'FIELD',
                description: 'A subset of AI',
                confidence: 0.90
            })
        """)
        print("✅ Entities inserted")
        
        # Create relation
        print("Creating relation...")
        conn.execute("""
            MATCH (a:Entity {id: 'e1'}), (b:Entity {id: 'e2'})
            CREATE (a)-[:RELATION {type: 'USED_IN', weight: 0.8, confidence: 0.85}]->(b)
        """)
        print("✅ Relation created")
        
        # Query graph
        print("Querying graph...")
        result = conn.execute("""
            MATCH (a:Entity)-[r:RELATION]->(b:Entity)
            RETURN a.name, r.type, b.name, r.weight
        """)
        
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        
        print(f"✅ Found {len(rows)} relations:")
        for row in rows:
            print(f"   - {row[0]} --[{row[1]}]--> {row[2]} (weight: {row[3]})")
        
        # Clean up
        shutil.rmtree(test_db)
        
        print("✅ Test 4 PASSED: KG operations work")
        return True
        
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """اجرای تمام تست‌ها"""
    print("\n" + "="*60)
    print("KuzuDB Integration Tests")
    print("="*60)
    
    tests = [
        ("Kuzu Availability", test_kuzu_availability),
        ("Direct Connection", test_direct_kuzu_connection),
        ("Wrapper Connection", test_wrapper_connection),
        ("Knowledge Graph Operations", test_knowledge_graph_operations),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASSED" if p else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

