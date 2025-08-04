from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Set

from core.database import get_db
from core.models import Document, DocumentResponse
from services.knowledge_graph import KnowledgeGraph
from core.config import index_endpoint, VERTEX_AI_DEPLOYED_INDEX_ID

router = APIRouter()
GRAPH_DIR = "knowledge_graphs"

@router.get("/documents/{client_id}", response_model=List[DocumentResponse])
def get_client_documents(client_id: str, db: Session = Depends(get_db)):
    """Returns a list of all documents for a given client."""
    documents = db.query(Document).filter(Document.client_id == client_id).all()
    if not documents:
        raise HTTPException(status_code=404, detail=f"No documents found for client ID: {client_id}")
    return documents
    return documents

@router.get("/query/")
async def query_documents(
    query: str,
    doc_ids: List[str] = Query(..., description="A list of one or more document IDs to search within."),
    db: Session = Depends(get_db)
):
    """Queries paragraphs and knowledge graphs across one or more specified documents."""
    
    # Construct the filter for Vertex AI based on the provided document IDs
    doc_id_filter = [{"namespace": "doc_id", "allow": doc_ids}]

    # 1. Query Vertex AI for relevant paragraphs across all specified documents
    paragraph_response = index_endpoint.find_neighbors(
        queries=[query],
        deployed_index_id=VERTEX_AI_DEPLOYED_INDEX_ID,
        num_neighbors=5,
        filter=doc_id_filter + [{"namespace": "type", "allow": ["paragraph"]}]
    )

    # 2. Query Vertex AI for relevant knowledge graph edges across all specified documents
    edge_response = index_endpoint.find_neighbors(
        queries=[query],
        deployed_index_id=VERTEX_AI_DEPLOYED_INDEX_ID,
        num_neighbors=5,
        filter=doc_id_filter + [{"namespace": "type", "allow": ["knowledge_graph_edge"]}]
    )

    # 3. Load the necessary knowledge graphs and expand context
    graph_context = ""
    loaded_graphs: Dict[str, KnowledgeGraph] = {}
    
    if edge_response and edge_response[0]:
        initial_nodes_by_doc: Dict[str, Set[str]] = {}
        
        # Group initial nodes by their document ID
        for match in edge_response[0]:
            # Metadata is not directly available in the response, so we parse from the ID
            # Assuming ID format is "{doc_id}_edge_{edge.id}"
            matched_doc_id = match.id.split('_edge_')[0]
            
            # We need the full edge info to get source/target nodes, which requires loading the graph
            if matched_doc_id not in loaded_graphs:
                graph = KnowledgeGraph.load(matched_doc_id, GRAPH_DIR)
                if graph: loaded_graphs[matched_doc_id] = graph
            
            # Find the edge in the loaded graph
            if matched_doc_id in loaded_graphs:
                edge_id = match.id.split('_edge_')[-1]
                retrieved_edge = next((e for e in loaded_graphs[matched_doc_id].edges if e.id == edge_id), None)
                if retrieved_edge:
                    if matched_doc_id not in initial_nodes_by_doc:
                        initial_nodes_by_doc[matched_doc_id] = set()
                    initial_nodes_by_doc[matched_doc_id].add(retrieved_edge.source.id)
                    initial_nodes_by_doc[matched_doc_id].add(retrieved_edge.target.id)

        # Expand context for each document's graph
        contexts = []
        for doc_id, node_ids in initial_nodes_by_doc.items():
            graph = loaded_graphs[doc_id]
            contexts.append(graph.expand_context_from_nodes(node_ids, depth=2))
        
        graph_context = "\n\n---\n\n".join(contexts)

    # 4. Combine and return results
    return {
        "query": query,
        "document_ids_queried": doc_ids,
        "retrieved_graph_context": graph_context,
        "retrieved_paragraphs": paragraph_response[0] if paragraph_response else []
    }
