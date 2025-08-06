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
    def __init__(self, name: str, page_no: Any):
        # A deterministic ID based on type and name ensures nodes are unique
        self.id = str(uuid.uuid4())
        self.name: str = name
        self.edges = []
        self.page_no: Any = page_no

    def add_edge(self, edge):
        """Adds a connected edge to this node's record."""
        self.edges.append(edge)

class Edge:
    """Represents a directed edge (relationship) in the knowledge graph."""
    def __init__(self, source: Node, target: Node, relation: str, page_no: Any):
        self.id: str = str(uuid.uuid4())
        self.source = source
        self.target = target
        self.relation = relation
        self.page_no = page_no

    def __repr__(self):
        # A more concise representation for the graph's __repr__ method.
        return f"--[{self.relation}]--> {self.target.name}"
    
    def to_sentence(self):
        """Converts the triplet into a natural language sentence."""
        return f"{self.source.name} {self.relation} {self.target.name}."
    

class KnowledgeGraph:
    """Manages the collection of nodes, edges, and interaction with the vector DB."""
    def __init__(self, doc_id: str, filename: str):
        self.doc_id = doc_id
        self.filename = filename
        self.nodes: Dict[str, Node] = {}
        self.node_name_map = {}
        self.edges: Dict[str, Edge] = {}

    def __repr__(self):
        """Provides a structured string representation of the entire graph."""
        if not self.nodes:
            return "KnowledgeGraph(empty)"
        
        representation = "KnowledgeGraph:\n"
        # Iterate through each node in the graph
        for node in self.node_name_map.values():
            # Check if there are any outgoing edges from this node
            outgoing_edges = [edge for edge in node.edges if edge.source == node]
            if outgoing_edges:
                representation += f"  - Node: {node.name}\n"
                for edge in outgoing_edges:
                    representation += f"    {edge}\n"
        return representation
    
    def get_or_create_node(self, name: str, page_no: Any) -> Node:
        """Adds a node if it's new, or returns the existing one."""
        if name not in self.node_name_map:
            node = Node(name, page_no)
            self.nodes[node.id] = node
            self.node_name_map[name] = node
            
        return self.node_name_map[name]

    def add_relation(self, relation_data: Dict[str, Any]):
        """Adds an edge to the graph based on the extracted relation data."""
        source_name = relation_data.get("subject", "")
        target_name = relation_data.get("object", "")
        relation = relation_data.get("predicate", "")
        page_no = relation_data.get("page_no", "unknown")
        
        if not all([source_name, target_name, relation]):
            return

        source_node = self.get_or_create_node(
            name=source_name, page_no=page_no
        )
        target_node = self.get_or_create_node(
            name=target_name, page_no=page_no
        )
        
        edge = Edge(source_node, target_node, relation, page_no)
        self.edges[edge.id] = edge
        
        source_node.add_edge(edge)
        target_node.add_edge(edge)
        
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Retrieves a node from the graph by its unique ID."""
        return self.nodes.get(node_id)

    def get_node_by_name(self, node_name: str) -> Optional[Node]:
        """Retrieves a node from the graph by its name."""
        return self.node_name_map.get(node_name)
        
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
    
    def get_context_from_nodes(self, nodes):
        """
        Retrieves all unique triplet sentences connected to a list of nodes.
        """
        context_triplets = set()
        for node in nodes:
            for edge in node.edges:
                context_triplets.add(edge.to_sentence())
        return list(context_triplets)
    
    def get_subgraph_context_from_edges(self, seed_edge_ids):
        """
        Traverses to depth 2 from seed edges and returns a structured string
        representation of the resulting subgraph, suitable for LLM context.
        """
        # 1. Find the initial set of nodes connected by the seed edges.
        nodes_to_expand = set()
        for edge_id in seed_edge_ids:
            if edge_id in self.edges:
                edge = self.edges[edge_id]
                nodes_to_expand.add(edge.source)
                nodes_to_expand.add(edge.target)

        # 2. Expand outwards twice (for depth 2)
        all_subgraph_nodes = set(nodes_to_expand)
        for _ in range(2): # Depth 1 and Depth 2 expansion
            if not nodes_to_expand:
                break
            
            current_expansion_set = set()
            for node in nodes_to_expand:
                for edge in node.edges:
                    neighbor = edge.source if edge.target == node else edge.target
                    if neighbor not in all_subgraph_nodes:
                        current_expansion_set.add(neighbor)
            
            all_subgraph_nodes.update(current_expansion_set)
            nodes_to_expand = current_expansion_set

        # 3. Build the structured string representation from the subgraph.
        representation = "Subgraph Context:\n"
        sorted_subgraph_nodes = sorted(list(all_subgraph_nodes), key=lambda n: n.name)

        for node in sorted_subgraph_nodes:
            # Find outgoing edges where the target is also in our subgraph
            outgoing_edges_in_subgraph = [
                edge for edge in node.edges 
                if edge.source == node and edge.target in all_subgraph_nodes
            ]
            if outgoing_edges_in_subgraph:
                representation += f"  - Node: {node.name}\n"
                for edge in sorted(outgoing_edges_in_subgraph, key=lambda e: e.relation):
                    representation += f"    {edge}\n"
        
        return representation if len(representation) > 18 else "Subgraph Context: No connected facts found."
    
    async def store_in_vector_db(self):
        """Generates embeddings for all edges and stores them in Vertex AI."""
        print("saving knowledge graph in vector db")
        if not vertex_ai_index:
            logger.warning("Vertex AI Vector Search not configured, skipping graph storage.")
            return
        if not self.edges:
            return

        try:
            texts_to_embed = [edge.to_sentence() for edge in self.edges.values()]
            # 2. Get all embeddings in a single, batched API call with retries
            # logger.info(f"Requesting embeddings for {len(texts_to_embed)} graph edges...")
            print(f"Requesting embeddings for {len(texts_to_embed)} graph edges...")
            max_retries = 5
            delay = 1.0
            embeddings = []
            for attempt in range(max_retries):
                try:
                    embeddings = embedding_model.get_embeddings(texts_to_embed)
                    break # Success
                except ResourceExhausted as e:
                    if attempt < max_retries - 1:
                        # logger.warning(f"Quota exceeded. Retrying in {delay:.2f} seconds...")
                        print(f"Quota exceeded. Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.error("Max retries reached for embedding generation. Failing.")
                        raise e
            
            # 3. Prepare datapoints for upserting
            datapoints = []
            for i, (edge_id, edge) in enumerate(self.edges.items()):
                restricts = [
                    {"namespace": "doc_id", "allow_list": [self.doc_id]},
                    {"namespace": "type", "allow_list": ["knowledge_graph_edge"]},
                    {"namespace": "edge_id", "allow_list": [edge_id]},
                    {"namespace": "source_node_id", "allow_list": [edge.source.id]},
                    {"namespace": "target_node_id", "allow_list": [edge.target.id]}
                ]
                datapoints.append({
                    "datapoint_id": f"{self.doc_id}_edge_{edge_id}",
                    "feature_vector": embeddings[i].values,
                    "restricts": restricts
                })
                
            #4. Upsert all datapoints in batches to avoid large requests
            if datapoints:
                # Upsert in batches to avoid large requests
                for i in range(0, len(datapoints), 100):
                    batch = datapoints[i:i+100]
                    vertex_ai_index.upsert_datapoints(datapoints=batch)
                logger.info(f"Upserted {len(datapoints)} graph relationships to Vertex AI for doc_id: {self.doc_id}")
        
        except Exception as e:
            logger.error(f"Failed to store graph relationships in Vertex AI for doc_id {self.doc_id}: {e}")

    