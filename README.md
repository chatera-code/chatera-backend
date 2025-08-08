# Chatera Backend: AI-Powered RAG and SQL Agent

This project is a sophisticated backend system designed to provide intelligent answers from both unstructured documents and structured databases. It is built on a Python microservice architecture using FastAPI and leverages Google's Gemini models for its AI capabilities.

The system is composed of three primary services that work together to handle document ingestion, user chat sessions, and advanced SQL querying.

## 🏛️ System Architecture

The application is designed as a set of containerized microservices orchestrated by Docker Compose for easy local development and scalability.



[Image of a microservice architecture diagram]


* **`main_app`**: The primary user-facing service. It handles all direct interactions with the client, including managing user sessions, chat history, document metadata, and real-time WebSocket communication. It acts as the central orchestrator.
* **`ingestion_service`**: A dedicated background worker responsible for handling file uploads and processing. It takes raw PDF documents, chunks them, extracts structured data (paragraphs, tables, relationships), builds knowledge graphs, and stores the processed data in the appropriate databases.
* **`sql_agent_service`**: A powerful, internal AI agent. When a user's query is determined to be about structured data, this service is called to autonomously explore SQL databases, construct and validate queries, and generate a natural-language answer based on the retrieved data.

---

## ✨ Key Features

* **Hybrid RAG & SQL**: Can answer questions from both uploaded PDF documents and structured SQL databases.
* **Automated Document Processing**: The ingestion pipeline automatically extracts paragraphs, tables, and relationships to build a comprehensive knowledge base.
* **Knowledge Graph Creation**: Builds and utilizes knowledge graphs to understand the relationships between entities in your documents.
* **Intelligent SQL Agent**: A Gemini-powered agent that can understand natural language, explore database schemas, write and validate its own SQL queries, and synthesize answers.
* **Real-time Updates**: Uses WebSockets to provide the front end with live progress updates during document ingestion and agent operation.
* **Microservice Architecture**: A scalable and maintainable design where each component runs as an independent, containerized service.

---

## 🚀 Getting Started (Local Development)

This guide will walk you through setting up and running the entire application on your local machine using Docker.

### Prerequisites

1.  **Docker Desktop**: You must have Docker and Docker Compose installed. Download it from the [official Docker website](https://www.docker.com/products/docker-desktop/).
2.  **Google Cloud Account**:
    * A Google Cloud Project with the **Vertex AI API** enabled.
    * A **Service Account** with the necessary permissions (`Vertex AI User`, `Cloud SQL Client`).
    * The **JSON key file** for this service account downloaded to your machine.
3.  **Environment File**:
    * You will need a `.env` file in the root of the project to store your credentials and configuration.

### Setup Instructions

1.  **Clone the Repository** (if applicable)
    ```bash
    git clone <your-repo-url>
    cd chatera-backend
    ```

2.  **Create the Environment File**:
    * In the root of the project, create a file named `.env`.
    * Add your credentials and configuration to this file. It should look like this:

    ```env
    # GCP Credentials & Configuration
    GCP_PROJECT_ID="your-gcp-project-id"
    GCP_REGION="your-gcp-region" # e.g., us-central1
    
    # Cloud SQL Credentials (for the SQL Agent)
    INSTANCE_CONNECTION_NAME="your-project:your-region:your-instance-name"
    DB_USER="your-db-user"
    DB_PASS="your-db-password"
    
    # Internal Service URLs (used by docker-compose, usually no change needed)
    MAIN_APP_NOTIFY_URL="http://main_app:8000/internal/notify"
    SQL_AGENT_URL="http://sql_agent_service:8000/agent/generate-sql"
    ```

3.  **Update `docker-compose.yaml`**:
    * Open the `docker-compose.yaml` file.
    * Find every line that says `- /path/to/your/service-account-key.json:/app/gcp_creds.json:ro`.
    * Replace `/path/to/your/service-account-key.json` with the actual, absolute path to the JSON key file you downloaded from Google Cloud.

### Running the Application

1.  **Initialize the Database**: The first time you run the application, the `db-init` service will automatically create the `rag_app.db` file and all necessary tables. This is handled by Docker Compose.

2.  **Build and Start the Services**: Open a terminal in the project root and run:
    ```bash
    docker-compose up --build
    ```
    * `--build`: This flag tells Docker to rebuild the images if any code or dependencies have changed.

3.  **Verify the Services**:
    * You will see logs from all services streamed to your terminal.
    * Once you see `Uvicorn running...` for `main_app`, `ingestion_service`, and `sql_agent_service`, the system is ready.
    * You can access the API documentation for the user-facing services in your browser:
        * **Main App**: `http://localhost:8000/docs`
        * **Ingestion Service**: `http://localhost:8002/docs`

4.  **Stopping the Application**:
    * To stop all running services, press `Ctrl + C` in the terminal where they are running.
    * To clean up and remove the containers and network, run:
    ```bash
    docker-compose down
    ```

---

## 🛠️ Project Structure

* **`/main_app`**: The primary user-facing service.
* **`/ingestion_service`**: Handles all file upload and processing tasks.
* **`/sql_agent_service`**: The internal AI agent for database queries.
* **`/data`**: A shared directory (managed by Docker volumes) that holds the SQLite database, uploaded files, and generated knowledge graphs.
* **`docker-compose.yaml`**: The master file that defines and orchestrates all the services.
* **`init_db.py`**: A one-time script, run by `docker-compose`, to initialize the database schema.
