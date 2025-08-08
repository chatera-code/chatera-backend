from pydantic import BaseModel, Field
from typing import List, Dict, Any

class SQLQueryRequest(BaseModel):
    natural_language_query: str
    databases: List[str] = Field(description="A list of database names within the Cloud SQL instance to search.")
    channel_id: str

class SQLQueryResponse(BaseModel):
    status: str # Can be "success" or "error"
    # The final_query field is no longer needed.
    final_response: str # This will always contain the agent's direct message to the user.
