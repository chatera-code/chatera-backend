import uuid, os
from typing import Dict, Any, Optional, List, Set
from google.cloud import aiplatform

from .utils import sanitize_name
from core.config import index_endpoint, VERTEX_AI_INDEX_ID, GCP_PROJECT_ID, GCP_REGION

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

    def get_or_create_node(self, name: str, node_type: str, attributes: Dict[str, Any], page_no: Any) -> Node:
        """Requirement 1 & 2: Adds a node if it's new, or returns the existing one."""
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
            name=source_data.get("name"), node_type=source_data.get("type"),
            attributes=source_data.get("attributes", {}), page_no=page_no
        )
        target_node = self.get_or_create_node(
            name=target_data.get("name"), node_type=target_data.get("type"),
            attributes=target_data.get("attributes", {}), page_no=page_no
        )
        
        edge = Edge(source_node, target_node, relation, comment, page_no)
        self.edges.append(edge)

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Retrieves a node from the graph by its unique ID."""
        return self.nodes.get(node_id)

    def store_in_vector_db(self):
        """Requirement 3: Generates embeddings for all edges and stores them in Vertex AI."""
        if not index_endpoint:
            print("Vertex AI Vector Search not configured, skipping graph storage.")
            return
        if not self.edges:
            return

        model = aiplatform.TextEmbeddingModel.from_pretrained("text-embedding-004")
        datapoints = []
        for edge in self.edges:
            embedding_text = repr(edge)
            embedding = model.get_embeddings([embedding_text])[0].values
            
            restricts = [
                {"namespace": "doc_id", "allow": [self.doc_id]},
                {"namespace": "type", "allow": ["knowledge_graph_edge"]},
                {"namespace": "edge_id", "allow": [edge.id]},
                {"namespace": "source_node_id", "allow": [edge.source.id]},
                {"namespace": "target_node_id", "allow": [edge.target.id]}
            ]
            datapoints.append({
                "datapoint_id": f"{self.doc_id}_edge_{edge.id}",
                "feature_vector": embedding,
                "restricts": restricts
            })
    
        if datapoints:
            index_endpoint.upsert_datapoints(index=VERTEX_AI_INDEX_ID, datapoints=datapoints)
            print(f"Upserted {len(datapoints)} graph relationships to Vertex AI for doc_id: {self.doc_id}")

    def _build_adjacency_list(self):
        """Builds an adjacency list for efficient graph traversal."""
        self._adjacency_list = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            self._adjacency_list[edge.source.id].append(edge)
            self._adjacency_list[edge.target.id].append(edge)

    def query_and_expand(self, query: str, depth: int = 2, similarity_threshold: float = 0.7, num_neighbors: int = 5) -> str:
        """Requirement 4: Queries the vector DB and recursively expands the context."""
        if not index_endpoint:
            return "Vector search is not configured."
        if self._adjacency_list is None:
            self._build_adjacency_list()

        # 1. Query Vertex AI to find the most similar edge(s)
        response = index_endpoint.find_neighbors(
            queries=[query],
            deployed_index_id=os.getenv("VERTEX_AI_DEPLOYED_INDEX_ID"), # Assumes a deployed index ID is in env
            num_neighbors=num_neighbors
        )

        # 2. Gather initial nodes from relevant edges
        initial_node_ids: Set[str] = set()
        retrieved_edges: Dict[str, Edge] = {edge.id: edge for edge in self.edges}
        
        if not response or not response[0]:
            return "No relevant information found in the knowledge graph."

        for match in response[0]:
            if match.distance >= similarity_threshold:
                edge_id = match.id.split('_edge_')[-1]
                edge = retrieved_edges.get(edge_id)
                if edge:
                    initial_node_ids.add(edge.source.id)
                    initial_node_ids.add(edge.target.id)

        # 3. Recursively find all connected edges up to the specified depth
        context_edges: Set[Edge] = set()
        nodes_to_visit: Set[str] = initial_node_ids
        
        for i in range(depth):
            if not nodes_to_visit:
                break
            
            next_nodes_to_visit: Set[str] = set()
            for node_id in nodes_to_visit:
                for edge in self._adjacency_list.get(node_id, []):
                    context_edges.add(edge)
                    next_nodes_to_visit.add(edge.source.id)
                    next_nodes_to_visit.add(edge.target.id)
            
            nodes_to_visit = next_nodes_to_visit - initial_node_ids
            initial_node_ids.update(nodes_to_visit)

        # 4. Return the combined context from all found edges
        if not context_edges:
            return "Found a potential match but could not expand context."
            
        return "\n".join([repr(edge) for edge in context_edges])
