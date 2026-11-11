"""
Neo4j Graph Ingestion Script
Project: Diseases Cause Oriented Analysis using RAG and Knowledge Graphs
PFA 5IADA - EMSI

Loads the targeted causal biomedical subgraph into Neo4j using the official
neo4j Python driver, defines constraints and indexes, and loads nodes and
typed relationships in optimized batches with UNWIND.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

# Try loading .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Graph.LoadNeo4j")


class Neo4jGraphLoader:
    """Ingests nodes and relationships into Neo4j with batching and constraints."""

    NODE_LABELS = [
        "Disease", "Gene", "Pathway", "Compound",
        "Symptom", "Anatomy", "SideEffect", "BiologicalProcess",
        "LifestyleCategory", "LifestyleFactor"
    ]

    RELATION_TYPES = [
        "TREATS", "PALLIATES", "ASSOCIATES_WITH", "DOWNREGULATES", "UPREGULATES",
        "MANIFESTS", "LOCALIZES_IN", "PARTICIPATES_IN", "BINDS", "INTERACTS_WITH",
        "REGULATES", "CAUSES", "RESEMBLES", "SUBCLASS_OF", "PROTECTS", "AGGRAVATES", "INDUCES"
    ]

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
        database: str = "neo4j",
        batch_size: int = 1000
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.batch_size = batch_size
        self.driver: Optional[Driver] = None

    def connect(self) -> bool:
        """Establishes connection to the Neo4j database."""
        logger.info(f"Connecting to Neo4j at {self.uri} (user: {self.user}, database: {self.database})...")
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j instance.")
            return True
        except (ServiceUnavailable, ConnectionRefusedError) as e:
            logger.error(f"Cannot connect to Neo4j at {self.uri}. Is Docker running? (Error: {e})")
            logger.error("Tip: Start Neo4j using 'docker-compose up -d neo4j'.")
            return False
        except AuthError as e:
            logger.error(f"Authentication failed for user '{self.user}'. Check your password. (Error: {e})")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while connecting to Neo4j: {e}")
            return False

    def close(self) -> None:
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver connection closed.")

    def create_constraints_and_indexes(self) -> None:
        """Creates uniqueness constraints and performance indexes for all node types."""
        logger.info("Creating uniqueness constraints and property indexes...")
        with self.driver.session(database=self.database) as session:
            for label in self.NODE_LABELS:
                # 1. Uniqueness constraint on ID
                constraint_query = (
                    f"CREATE CONSTRAINT constraint_{label.lower()}_id IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                )
                try:
                    session.run(constraint_query)
                    logger.debug(f"Constraint created for :{label}(id)")
                except Exception as e:
                    logger.warning(f"Could not create constraint for {label}: {e}")

                # 2. Performance index on name
                index_name_query = (
                    f"CREATE INDEX index_{label.lower()}_name IF NOT EXISTS "
                    f"FOR (n:{label}) ON (n.name)"
                )
                try:
                    session.run(index_name_query)
                except Exception as e:
                    logger.warning(f"Could not create index on {label}(name): {e}")

                # 3. Index on UMLS CUI if applicable
                index_cui_query = (
                    f"CREATE INDEX index_{label.lower()}_umls IF NOT EXISTS "
                    f"FOR (n:{label}) ON (n.umls_cui)"
                )
                try:
                    session.run(index_cui_query)
                except Exception:
                    pass

        logger.info("Constraints and indexes successfully verified.")

    def load_nodes(self, df_nodes: pd.DataFrame) -> int:
        """Loads nodes grouped by label using parameterized UNWIND batches."""
        logger.info(f"Loading {len(df_nodes):,} nodes into Neo4j in batches of {self.batch_size}...")
        total_loaded = 0

        # Replace NaN with empty string
        df_nodes = df_nodes.fillna("")

        with self.driver.session(database=self.database) as session:
            for label, group in df_nodes.groupby("type"):
                records = group.to_dict(orient="records")
                query = f"""
                UNWIND $batch AS row
                MERGE (n:{label} {{id: row.id}})
                SET n.name = row.name,
                    n.identifier_type = row.identifier_type,
                    n.external_id = row.external_id,
                    n.umls_cui = row.umls_cui,
                    n.mesh_id = row.mesh_id,
                    n.doid = row.doid
                """
                for i in range(0, len(records), self.batch_size):
                    batch = records[i:i + self.batch_size]
                    session.run(query, batch=batch)
                    total_loaded += len(batch)
                logger.info(f"  - Loaded {len(records):,} :{label} nodes.")

        logger.info(f"Total nodes loaded: {total_loaded:,}")
        return total_loaded

    def load_relationships(self, df_edges: pd.DataFrame) -> int:
        """Loads relationships grouped by typed relation using parameterized UNWIND batches."""
        logger.info(f"Loading {len(df_edges):,} relationships into Neo4j in batches of {self.batch_size}...")
        total_loaded = 0
        df_edges = df_edges.fillna("")

        with self.driver.session(database=self.database) as session:
            for rel_type, group in df_edges.groupby("relation"):
                records = group.to_dict(orient="records")
                # Parameterized Cypher query with concrete relationship type
                query = f"""
                UNWIND $batch AS row
                MATCH (a {{id: row.source_id}})
                MATCH (b {{id: row.target_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.metaedge = row.metaedge,
                    r.causal_type = row.causal_type,
                    r.hop = toInteger(row.hop),
                    r.source_dataset = row.source_dataset
                """
                for i in range(0, len(records), self.batch_size):
                    batch = records[i:i + self.batch_size]
                    session.run(query, batch=batch)
                    total_loaded += len(batch)
                logger.info(f"  - Loaded {len(records):,} [:{rel_type}] relationships.")

        logger.info(f"Total relationships loaded: {total_loaded:,}")
        return total_loaded

    def verify_ingestion(self) -> Dict[str, Any]:
        """Runs validation count queries and returns summary stats."""
        stats = {}
        logger.info("Verifying Neo4j graph contents...")
        with self.driver.session(database=self.database) as session:
            # Count nodes by label
            res_nodes = session.run("""
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS cnt', {})
                YIELD value
                RETURN label, value.cnt AS count
            """) if False else session.run("""
                MATCH (n) RETURN head(labels(n)) AS label, count(n) AS count ORDER BY count DESC
            """)
            stats["nodes_by_label"] = {record["label"]: record["count"] for record in res_nodes}

            # Count relationships by type
            res_rels = session.run("""
                MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count ORDER BY count DESC
            """)
            stats["rels_by_type"] = {record["rel_type"]: record["count"] for record in res_rels}

            # Quick causal test path count: Alzheimer -> Gene -> Pathway
            res_causal = session.run("""
                MATCH (d:Disease {name: "Alzheimer's disease"})-[:ASSOCIATES_WITH]->(g:Gene)-[:PARTICIPATES_IN]->(p:Pathway)
                RETURN count(DISTINCT p) AS distinct_pathways, count(DISTINCT g) AS distinct_genes
            """)
            record = res_causal.single()
            if record:
                stats["alzheimer_causal_pathways"] = record["distinct_pathways"]
                stats["alzheimer_causal_genes"] = record["distinct_genes"]

        logger.info(f"Graph verification stats: {stats}")
        return stats


def validate_local_dataset(nodes_csv: Path, edges_csv: Path) -> Dict[str, Any]:
    """Validates processed CSVs without requiring a live Neo4j instance."""
    logger.info("Running offline data integrity validation...")
    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edges_csv)

    node_ids = set(df_nodes["id"])
    missing_sources = set(df_edges["source_id"]) - node_ids
    missing_targets = set(df_edges["target_id"]) - node_ids

    is_valid = len(missing_sources) == 0 and len(missing_targets) == 0
    report = {
        "status": "VALID" if is_valid else "INVALID",
        "total_nodes": len(df_nodes),
        "total_edges": len(df_edges),
        "missing_source_ids": len(missing_sources),
        "missing_target_ids": len(missing_targets),
        "node_types": df_nodes["type"].value_counts().to_dict(),
        "relation_types": df_edges["relation"].value_counts().to_dict()
    }
    logger.info(f"Validation Report: Status={report['status']}, Nodes={report['total_nodes']:,}, Edges={report['total_edges']:,}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Ingest targeted subgraph into Neo4j.")
    parser.add_argument("--nodes-csv", type=str, default="data/processed/nodes_targeted.csv", help="Path to nodes CSV")
    parser.add_argument("--edges-csv", type=str, default="data/processed/edges_targeted.csv", help="Path to edges CSV")
    parser.add_argument("--uri", type=str, default=os.getenv("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j Bolt URI")
    parser.add_argument("--user", type=str, default=os.getenv("NEO4J_USER", "neo4j"), help="Neo4j username")
    parser.add_argument("--password", type=str, default=os.getenv("NEO4J_PASSWORD", "password123"), help="Neo4j password")
    parser.add_argument("--database", type=str, default=os.getenv("NEO4J_DATABASE", "neo4j"), help="Database name")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for UNWIND")
    parser.add_argument("--dry-run", action="store_true", help="Validate CSV data integrity without connecting to Neo4j")

    args = parser.parse_args()
    nodes_path = Path(args.nodes_csv)
    edges_path = Path(args.edges_csv)

    if not nodes_path.exists() or not edges_path.exists():
        logger.error(f"Processed CSV files not found. Run 'python src/etl/extract_subgraph.py' first.")
        sys.exit(1)

    # Always perform data integrity validation first
    val_report = validate_local_dataset(nodes_path, edges_path)
    if val_report["status"] != "VALID":
        logger.error(f"Integrity check failed: {val_report}")
        sys.exit(1)

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY-RUN COMPLETE: Data is 100% structurally valid for Neo4j.")
        print("=" * 60)
        return

    loader = Neo4jGraphLoader(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        batch_size=args.batch_size
    )

    if not loader.connect():
        logger.warning("Neo4j database is currently unreachable. You can launch it with:")
        logger.warning("  docker-compose up -d neo4j")
        logger.warning("Once the container is healthy, rerun this script to complete live ingestion.")
        return

    try:
        t0 = time.time()
        df_nodes = pd.read_csv(nodes_path)
        df_edges = pd.read_csv(edges_path)

        loader.create_constraints_and_indexes()
        n_nodes = loader.load_nodes(df_nodes)
        n_edges = loader.load_relationships(df_edges)
        stats = loader.verify_ingestion()
        elapsed = time.time() - t0

        print("\n" + "=" * 60)
        print("NEO4J INGESTION SUCCESSFULLY COMPLETED")
        print("=" * 60)
        print(f"Elapsed Time     : {elapsed:.2f} seconds")
        print(f"Nodes Ingested   : {n_nodes:,}")
        print(f"Edges Ingested   : {n_edges:,}")
        print(f"Verification     : {stats}")
        print("=" * 60)

    finally:
        loader.close()


if __name__ == "__main__":
    main()

