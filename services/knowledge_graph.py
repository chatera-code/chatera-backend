import uuid
import os
import pickle
from typing import Dict, Any, Optional, List, Set
from google.cloud import aiplatform
from google.api_core.exceptions import ResourceExhausted
import time
import logging
from .utils import sanitize_name, call_with_retry
from core.config import index_endpoint, embedding_model, VERTEX_AI_INDEX_ID, VERTEX_AI_DEPLOYED_INDEX_ID, vertex_ai_index

logger = logging.getLogger(__name__)

class Node:
    """Represents a single node in the knowledge graph."""
    def __init__(self, name: str, node_type: str, attributes: Dict[str, Any], page_no: Any):
        # A deterministic ID based on type and name ensures nodes are unique
        self.id: str = f"{node_type.lower()}_{sanitize_name(name).lower()}"
        self.name: str = name
        self.type: str = node_type
        self.attributes: Dict[str, Any] = attributes
        self.page_no: Any = page_no

    def __repr__(self) -> str:
        return f"Node(id={self.id}, name='{self.name}', type='{self.type}')"

class Edge:
    """Represents a directed edge (relationship) in the knowledge graph."""
    def __init__(self, source: Node, target: Node, relation: str, comment: str, page_no: Any):
        self.id: str = str(uuid.uuid4())
        self.source = source
        self.target = target
        self.relation = relation
        self.comment = comment
        self.page_no = page_no

    def __repr__(self) -> str:
        """Generates the string representation of the relationship for embedding."""
        return (f"Fact: The entity '{self.source.name}' ({self.source.type}) {self.relation} "
                f"the entity '{self.target.name}' ({self.target.type}). "
                f"Supporting context: {self.comment}")

class KnowledgeGraph:
    """Manages the collection of nodes, edges, and interaction with the vector DB."""
    def __init__(self, doc_id: str, filename: str):
        self.doc_id = doc_id
        self.filename = filename
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._adjacency_list: Optional[Dict[str, List[Edge]]] = None

    def __repr__(self) -> str:
        """Generates a node-centric string representation of the graph."""
        if not self.nodes:
            return "KnowledgeGraph is empty."
        
        output = ["Knowledge Graph Context:\n"]
        if self._adjacency_list is None:
            self._build_adjacency_list()

        for node_id, node in self.nodes.items():
            output.append(f"  - Node: {node.name} ({node.type}) [ID: {node.id}]")
            connections = self._adjacency_list.get(node_id, [])
            if connections:
                for edge in connections:
                    if edge.source.id == node_id:
                        output.append(f"    - {edge.relation} -> {edge.target.name} ({edge.target.type})")
                    else:
                        output.append(f"    - is {edge.relation} of <- {edge.source.name} ({edge.source.type})")
            else:
                output.append("    - (No direct connections in this context)")
        return "\n".join(output)
    
    def get_or_create_node(self, name: str, node_type: str, attributes: Dict[str, Any], page_no: Any) -> Node:
        """Adds a node if it's new, or returns the existing one."""
        node_id = f"{node_type.lower()}_{sanitize_name(name).lower()}"
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(name, node_type, attributes, page_no)
        return self.nodes[node_id]

    def add_relation(self, relation_data: Dict[str, Any], page_no: Any):
        """Adds an edge to the graph based on the extracted relation data."""
        source_data = relation_data.get("source", {})
        target_data = relation_data.get("target", {})
        relation = relation_data.get("relation")
        comment = relation_data.get("comment", "")

        if not all([source_data.get("name"), target_data.get("name"), relation]):
            return

        source_node = self.get_or_create_node(
            name=source_data.get("name"), node_type=source_data.get("type", "Unknown"),
            attributes=source_data.get("attributes", {}), page_no=page_no
        )
        target_node = self.get_or_create_node(
            name=target_data.get("name"), node_type=target_data.get("type", "Unknown"),
            attributes=target_data.get("attributes", {}), page_no=page_no
        )
        
        edge = Edge(source_node, target_node, relation, comment, page_no)
        self.edges.append(edge)
        self._adjacency_list = None # Invalidate adjacency list

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Retrieves a node from the graph by its unique ID."""
        return self.nodes.get(node_id)

    def save(self, directory: str):
        """Saves the entire KnowledgeGraph object to a file."""
        print(f"Saving knowledge graph for doc_id {self.doc_id} to {directory}")
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.doc_id}.graph")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Knowledge graph for doc_id {self.doc_id} saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save knowledge graph for doc_id {self.doc_id}: {e}")

    @staticmethod
    def load(doc_id: str, directory: str) -> Optional['KnowledgeGraph']:
        """Loads a KnowledgeGraph object from a file."""
        filepath = os.path.join(directory, f"{doc_id}.graph")
        if not os.path.exists(filepath):
            logger.warning(f"Knowledge graph file not found for doc_id {doc_id} at {filepath}")
            return None
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load knowledge graph for doc_id {doc_id}: {e}")
            return None
        
    async def store_in_vector_db(self):
        """Generates embeddings for all edges and stores them in Vertex AI."""
        if not vertex_ai_index:
            logger.warning("Vertex AI Vector Search not configured, skipping graph storage.")
            return
        if not self.edges:
            return

        try:
            # 1. Collect all edge representations into a single list for batching
            texts_to_embed = [repr(edge) for edge in self.edges]
            
            # 2. Get all embeddings in a single, batched API call with retries
            logger.info(f"Requesting embeddings for {len(texts_to_embed)} graph edges...")
            
            max_retries = 5
            delay = 1.0
            embeddings = []
            for attempt in range(max_retries):
                try:
                    embeddings = embedding_model.get_embeddings(texts_to_embed)
                    break # Success
                except ResourceExhausted as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Quota exceeded. Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.error("Max retries reached for embedding generation. Failing.")
                        raise e
            
            # 3. Prepare datapoints for upserting
            datapoints = []
            for i, edge in enumerate(self.edges):
                restricts = [
                    {"namespace": "doc_id", "allow_list": [self.doc_id]},
                    {"namespace": "type", "allow_list": ["knowledge_graph_edge"]},
                    {"namespace": "edge_id", "allow_list": [edge.id]},
                    {"namespace": "source_node_id", "allow_list": [edge.source.id]},
                    {"namespace": "target_node_id", "allow_list": [edge.target.id]}
                ]
                datapoints.append({
                    "datapoint_id": f"{self.doc_id}_edge_{edge.id}",
                    "feature_vector": embeddings[i].values,
                    "restricts": restricts
                })
                
            #4. Upsert all datapoints in batches to avoid large requests
            if datapoints:
                # Upsert in batches to avoid large requests
                for i in range(0, len(datapoints), 100):
                    batch = datapoints[i:i+100]
                    vertex_ai_index.upsert_datapoints(index=VERTEX_AI_INDEX_ID, datapoints=batch)
                logger.info(f"Upserted {len(datapoints)} graph relationships to Vertex AI for doc_id: {self.doc_id}")
        
        except Exception as e:
            logger.error(f"Failed to store graph relationships in Vertex AI for doc_id {self.doc_id}: {e}")

    def _build_adjacency_list(self):
        """Builds an adjacency list for efficient graph traversal."""
        self._adjacency_list = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            self._adjacency_list[edge.source.id].append(edge)
            self._adjacency_list[edge.target.id].append(edge)

    def expand_context_from_nodes(self, initial_node_ids: Set[str], depth: int = 2) -> str:
        """
        Recursively finds all connected edges from a set of starting nodes
        up to a specified depth and returns their string representations.
        """
        if self._adjacency_list is None:
            self._build_adjacency_list()

        context_edges: Set[Edge] = set()
        nodes_to_visit: Set[str] = set(initial_node_ids)
        visited_nodes: Set[str] = set()

        for i in range(depth):
            if not nodes_to_visit:
                break
            
            # Nodes to visit in the next level of the expansion
            next_nodes_to_visit: Set[str] = set()
            
            # Mark the current nodes as visited
            visited_nodes.update(nodes_to_visit)
            
            for node_id in nodes_to_visit:
                # Find all edges connected to the current node
                for edge in self._adjacency_list.get(node_id, []):
                    context_edges.add(edge)
                    # Add the neighbors to the set for the next level of traversal
                    if edge.source.id not in visited_nodes:
                        next_nodes_to_visit.add(edge.source.id)
                    if edge.target.id not in visited_nodes:
                        next_nodes_to_visit.add(edge.target.id)
            
            nodes_to_visit = next_nodes_to_visit

        if not context_edges:
            return "Could not expand context from the provided nodes."
            
        return "\n".join([repr(edge) for edge in context_edges])

    def query_and_expand(self, query: str, depth: int = 2, similarity_threshold: float = 0.7, num_neighbors: int = 5) -> str:
        """Queries the vector DB for relevant edges and expands the context."""
        if not index_endpoint or not VERTEX_AI_DEPLOYED_INDEX_ID:
            logger.warning("Vector search is not configured, cannot perform query_and_expand.")
            return "Vector search is not configured."
        if self._adjacency_list is None:
            self._build_adjacency_list()

        try:
            # 1. Query Vertex AI to find the most similar edge(s)
            response = index_endpoint.find_neighbors(
                queries=[query],
                deployed_index_id=VERTEX_AI_DEPLOYED_INDEX_ID,
                num_neighbors=num_neighbors
            )
        except Exception as e:
            logger.error(f"Vertex AI find_neighbors call failed: {e}")
            return "Error querying the vector database."

        # 2. Gather initial nodes from relevant edges
        initial_node_ids: Set[str] = set()
        retrieved_edges: Dict[str, Edge] = {edge.id: edge for edge in self.edges}
        
        if not response or not response[0]:
            return "No relevant information found in the knowledge graph."

        for match in response[0]:
            if match.distance >= similarity_threshold:
                # ID format is assumed to be "{doc_id}_edge_{edge.id}"
                edge_id = match.id.split('_edge_')[-1]
                edge = retrieved_edges.get(edge_id)
                if edge:
                    initial_node_ids.add(edge.source.id)
                    initial_node_ids.add(edge.target.id)
        
        if not initial_node_ids:
            return "Found potential matches but could not identify initial nodes for context expansion."

        # 3. Recursively expand the context from these initial nodes
        return self.expand_context_from_nodes(initial_node_ids, depth)