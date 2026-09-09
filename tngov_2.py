import json
import os
from neo4j import GraphDatabase

# Configuration Settings (Adjust with your database credentials)
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://a249ad00.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USER", "a249ad00")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "9R_2TZNIiuFkh-DdoCX8Zk1VTZ4ncQ1zCJV-_sEOi34")


class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def setup_constraints(self):
        """Setup unique constraints and indexes to optimize lookup performance."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Scheme) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Beneficiary) REQUIRE b.type IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (doc:Document) REQUIRE doc.name IS UNIQUE"
        ]
        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)
        print("Database indexes and constraints set up successfully.")

    def ingest_scheme(self, record):
        """Cypher query execution to map JSON scheme attributes into Graph entities."""
        cypher_query = """
        // 1. Create or Match Scheme Node
        MERGE (s:Scheme {name: $scheme_name})
        ON CREATE SET 
            s.source_url = $source_url,
            s.eligibility_criteria = $eligibility,
            s.subsidy_details = $subsidy_details,
            s.application_process = $application_process

        // 2. Link Department Node
        MERGE (d:Department {name: $department})
        MERGE (s)-[:MANAGED_BY]->(d)

        // 3. Link Target Beneficiary Node
        MERGE (b:Beneficiary {type: $beneficiary_type})
        MERGE (s)-[:APPLIES_TO]->(b)

        // 4. Extract and Link Required Documents
        WITH s, $documents_required AS docs_str
        UNWIND split(docs_str, ',') AS doc_item
        WITH s, trim(doc_item) AS doc_name
        WHERE doc_name <> '' AND doc_name <> 'N/A'
        MERGE (doc:Document {name: doc_name})
        MERGE (s)-[:REQUIRES_DOC]->(doc)
        """

        entities = record["structured_entities"]
        params = {
            "scheme_name": entities.get("scheme_name"),
            "source_url": record.get("source_url"),
            "eligibility": entities.get("eligibility"),
            "subsidy_details": entities.get("subsidy_details"),
            "application_process": entities.get("application_process"),
            "department": entities.get("department", "Agriculture Department"),
            "beneficiary_type": entities.get("beneficiary_type", "Farmers"),
            "documents_required": entities.get("documents_required", "")
        }

        with self.driver.session() as session:
            session.run(cypher_query, **params)


def main():
    json_path = "tn_agriculture_schemes.json"

    if not os.path.exists(json_path):
        print(f"Error: Could not find '{json_path}'. Make sure Step 2 was executed successfully.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Connecting to Neo4j database at: {NEO4J_URI}...")
    builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    try:
        builder.setup_constraints()
        print(f"Ingesting {len(data)} schemes into Knowledge Graph...")

        for idx, item in enumerate(data):
            builder.ingest_scheme(item)
            print(f"[{idx + 1}/{len(data)}] Ingested Scheme: {item['structured_entities']['scheme_name']}")

        print("\nGraph Ingestion Complete!")

    finally:
        builder.close()

#map JSON scheme attributes into Graph entities
if __name__ == "__main__":
    main()