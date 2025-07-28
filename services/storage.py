import os
from typing import List, Dict
from sqlalchemy import text
from google.cloud import aiplatform

from .utils import sanitize_name
from core.config import engine_mysql, index_endpoint, GCP_PROJECT_ID, GCP_REGION, VERTEX_AI_INDEX_ID

def store_tables_in_mysql(doc_id: str, filename: str, tables: List[Dict]):
    """Dynamically creates a database per document and populates tables in it."""
    if not engine_mysql:
        print("MySQL not configured, skipping table storage.")
        return

    # Create a sanitized database name from the document filename (without extension)
    db_name_raw = os.path.splitext(filename)[0]
    db_name = sanitize_name(db_name_raw)

    with engine_mysql.connect() as connection:
        try:
            # Create the database for the document if it doesn't exist.
            # The user needs CREATE DATABASE privileges.
            connection.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
            # Switch the connection's context to the new database.
            connection.execute(text(f"USE `{db_name}`"))
            print(f"Switched to database: `{db_name}`")

            # Now, proceed with table creation within this database
            for i, table_data in enumerate(tables):
                # Sanitize names for SQL
                clean_doc_id = sanitize_name(doc_id)
                page_no = table_data.get('page_no', 'unknown')
                page_str = f"p{page_no}" if isinstance(page_no, int) else f"p{page_no[0]}"
                # Table name is now simpler as it's inside a document-specific DB
                table_name = f"table_{i+1}_{page_str}"
                
                columns = table_data.get("table_content", [])
                if not columns: continue
                
                create_stmt = f"CREATE TABLE IF NOT EXISTS `{table_name}` (id INT AUTO_INCREMENT PRIMARY KEY, "
                column_defs = [f"`{sanitize_name(c.get('column_name', f'col_{i}'))}` TEXT" for c in columns]
                create_stmt += ", ".join(column_defs) + ");"
                
                connection.execute(text(create_stmt))
                
                max_rows = max(len(col.get("column_value", [])) for col in columns) if columns else 0
                for row_idx in range(max_rows):
                    col_names = [f"`{sanitize_name(c.get('column_name'))}`" for c in columns]
                    insert_stmt = f"INSERT INTO `{table_name}` ({', '.join(col_names)}) VALUES ("
                    values = [f"'{str(col.get('column_value', [])[row_idx] if row_idx < len(col.get('column_value', [])) else None).replace('\'', '\\\'')}'" for col in columns]
                    insert_stmt += ", ".join(values) + ");"
                    connection.execute(text(insert_stmt))
            
            connection.commit()
            print(f"Successfully stored {len(tables)} tables in MySQL database: {db_name}")

        except Exception as e:
            print(f"An error occurred during MySQL operations for database {db_name}: {e}")


def store_paragraphs_in_vertex_ai(doc_id: str, filename: str, paragraphs: List[Dict]):
    """Generates embeddings and stores paragraphs in Vertex AI Vector Search."""
    if not index_endpoint:
        print("Vertex AI Vector Search not configured, skipping vector storage.")
        return
    if not paragraphs:
        return

    model = aiplatform.TextEmbeddingModel.from_pretrained("text-embedding-004")
    datapoints = []
    for i, para_data in enumerate(paragraphs):
        embedding = model.get_embeddings([para_data["text"]])[0].values
        restricts = [
            {"namespace": "doc_id", "allow": [doc_id]},
            {"namespace": "filename", "allow": [filename]},
            {"namespace": "page_no", "allow": [str(para_data.get("page_no", "unknown"))]}
        ]
        datapoints.append({
            "datapoint_id": f"{doc_id}_para_{i}",
            "feature_vector": embedding,
            "restricts": restricts
        })
    
    if datapoints:
        index_endpoint.upsert_datapoints(index=VERTEX_AI_INDEX_ID, datapoints=datapoints)
        print(f"Upserted {len(datapoints)} paragraphs to Vertex AI for doc_id: {doc_id}")