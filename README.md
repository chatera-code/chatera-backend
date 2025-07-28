# Cloud-Native RAG Ingestion Pipeline

This project provides a robust, **cloud-native backend service** for ingesting PDF documents, extracting structured information using the **Google Gemini API**, and populating both a **relational database** and a **vector store** to power a sophisticated Retrieval-Augmented Generation (RAG) application.

---

## Overview

At its core, this application automates the **ingestion** part of a RAG pipeline. It takes a raw PDF document, analyzes its content page by page, and intelligently extracts three types of information:

- **Paragraphs**: Plain text content, which is vectorized for semantic search.
- **Tables**: Structured data, which is stored in a relational database for precise lookups.
- **Knowledge Graph Relations**: Entities and their relationships, which are vectorized and stored to enable complex, context-aware queries.

> **Note**: This initial version of the project focuses exclusively on the **ingestion pipeline**. The retrieval API is planned for future development.

---

## Ingestion Workflow

The data ingestion and processing pipeline follows these steps:

### 1. **User & API Layer**
➡️ User uploads a PDF document via a **POST** request to the `/upload/` endpoint.  
➡️ The API immediately creates an initial record for the document in a **SQLite database** with a status of `received`.  
➡️ An **asynchronous background task**, `process_document`, is triggered to handle the heavy lifting.

### 2. **Document Processing Service**
📄 The `process_document` task begins.  
✂️ It first splits the uploaded PDF into smaller, **10-page chunks**.

### 3. **Chunk Processing Loop (Repeats for each chunk)**
🧠 **Build Gemini Prompt**: A detailed prompt is constructed, which includes the instructions and any knowledge graph relations extracted from previous chunks (`knowledge_context`).  
🚀 **Call Gemini API**: The PDF chunk and the prompt are sent to the **Gemini 1.5 Pro API**.  
📥 **Parse JSON Response**: The application receives a structured **JSON object** from Gemini containing **paragraphs, tables, and relations**.

### 4. **Data Storage & Graph Building (Inside the loop)**
- **For each Paragraph**:  
  The text is sent to **Vertex AI Vector Search** to be stored as a vector embedding with its metadata (`doc_id`, `page_no`).
- **For each Table**:  
  A new database is created in **Cloud SQL for MySQL** named after the document.  
  A new table is created within that database, and the extracted tabular data is inserted.
- **For each Relation**:  
  The relation is added to an **in-memory Knowledge Graph object**, creating or linking nodes and edges.

### 5. **Finalization (After all chunks are processed)**
🧠 **Store Knowledge Graph**: The complete, in-memory Knowledge Graph is processed.  
The string representation of each edge (relationship) is converted into a **vector embedding**.  
These embeddings are stored in **Vertex AI Vector Search** with their metadata (source/target node IDs).  
✅ **Update Status**: The document's status is updated to `completed` in the main **SQLite database**.

---

## Tech Stack

- **Backend Framework**: FastAPI  
- **Language**: Python 3.9+  
- **AI/ML Models**: Google Gemini 1.5 Pro, Vertex AI Text Embedding Model  
- **Vector Store**: Google Cloud Vertex AI Vector Search  
- **Relational Database**: Google Cloud SQL for MySQL (with local MySQL support for development)  
- **Metadata Database**: SQLite  
- **Real-time Updates**: WebSockets  

---

## Project Structure

```
rag_project/
│
├── api/
│   ├── __init__.py
│   ├── upload.py           # Handles the /upload endpoint
│   └── websockets.py       # Handles the WebSocket connection
│
├── core/
│   ├── __init__.py
│   ├── config.py           # Manages environment variables and client initializations
│   ├── database.py         # Handles SQLite session management
│   └── models.py           # Defines SQLAlchemy and Pydantic models
│
├── services/
│   ├── __init__.py
│   ├── document_processor.py # Main business logic for processing a document
│   ├── knowledge_graph.py  # Classes for building and managing the knowledge graph
│   ├── storage.py          # Logic for storing data in MySQL and Vertex AI
│   └── utils.py            # Utility functions (e.g., PDF splitting)
│
├── main.py                 # Main FastAPI application entry point
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

---

## Setup and Installation

### 1. **Prerequisites**
- A **Google Cloud Platform (GCP)** project with billing enabled.
- The `gcloud` CLI installed and authenticated (`gcloud auth application-default login`).
- The following GCP APIs enabled:
  - **Vertex AI API**
  - **Cloud SQL Admin API**
- A provisioned **Vertex AI Vector Search Index and Endpoint**.
- A provisioned **Cloud SQL for MySQL instance** (for cloud deployment).
- A **local MySQL server** (for local development).

---

### 2. **Environment Variables**

Create a `.env` file in the project root (or set environment variables directly) with the following information:

```
# --- General GCP Config ---
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
GCP_PROJECT_ID="your-gcp-project-id"
GCP_REGION="us-central1" # The region for your GCP resources

# --- Vertex AI Config ---
VERTEX_AI_INDEX_ID="your-vertex-ai-index-id"
VERTEX_AI_INDEX_ENDPOINT_ID="your-vertex-ai-endpoint-id"
VERTEX_AI_DEPLOYED_INDEX_ID="your-deployed-index-id" # For querying

# --- Database Config ---
# Set USE_CLOUD_SQL to "true" for cloud, "false" for local
USE_CLOUD_SQL="false"

# If using Cloud SQL (USE_CLOUD_SQL="true")
INSTANCE_CONNECTION_NAME="your-gcp-project-id:us-central1:your-instance-name"
DB_USER="your-cloud-sql-user"
DB_PASS="your-cloud-sql-password"

# If using Local MySQL (USE_CLOUD_SQL="false")
# The URL should point to the server, not a specific DB, to allow dynamic DB creation
MYSQL_URL="mysql+pymysql://your_local_user:your_local_password@localhost/"
```

---

### 3. **Installation**

Clone the repository and install the required dependencies.

```bash
# Clone the repo (replace with your actual repo URL)
git clone https://github.com/your-username/rag-project.git
cd rag-project

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 4. **Running the Application**

Start the FastAPI server using **Uvicorn**:

```bash
uvicorn main:app --reload
```

The API will be available at:  
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Usage

Use a tool like **curl** to upload a PDF document to the `/upload/` endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/upload/?client_id=test_client_01" -F "file=@/path/to/your/document.pdf"
```

You can monitor the real-time progress of the ingestion by connecting a **WebSocket client** to:

```
ws://127.0.0.1:8000/ws/test_client_01
```

---

## Next Steps

- **Retrieval API**: Develop endpoints to query the stored data. This will involve:
  - Querying Vertex AI to find relevant paragraphs and knowledge graph edges.
  - Using the retrieved context to build a prompt for a generation model.
- **Error Handling**: Enhance error handling and add a retry mechanism for API calls.
- **Testing**: Add a comprehensive suite of unit and integration tests.

---
