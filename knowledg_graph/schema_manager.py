"""
Semantic Graph Schema:
Node labels: Entity, Concept, Document, Chunk, Source
Edge types:  RELATES_TO, PART_OF, MENTIONS, LINKED_TO,
             CONTRADICTS, SUPPORTS, DERIVED_FROM
"""

SCHEMA_QUERIES = [
    # Constraints
    "CREATE CONSTRAINT entity_id IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.id IS UNIQUE",

    "CREATE CONSTRAINT concept_id IF NOT EXISTS "
    "FOR (c:Concept) REQUIRE c.id IS UNIQUE",

    # Vector Index روی embedding ذخیره‌شده در Neo4j
    "CREATE VECTOR INDEX entity_embedding IF NOT EXISTS "
    "FOR (e:Entity) ON e.embedding "
    "OPTIONS {indexConfig: {`vector.dimensions`: 596, "
    "`vector.similarity_function`: 'cosine'}}",

    # Full-text index
    "CREATE FULLTEXT INDEX entity_text IF NOT EXISTS "
    "FOR (e:Entity) ON EACH [e.name, e.description]",
]


 