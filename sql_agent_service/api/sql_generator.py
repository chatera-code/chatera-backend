# In api/sql_generator.py

from fastapi import APIRouter, HTTPException
from core.models import SQLQueryRequest, SQLQueryResponse
from services.sql_agent.agent import run_sql_agent
import traceback

router = APIRouter()

@router.post(
    "/generate-sql",
    response_model=SQLQueryResponse,
    tags=["SQL Agent"],
    summary="Generate a validated SQL query from natural language"
)
async def generate_sql_endpoint(request: SQLQueryRequest):
    """
    Takes a natural language query and orchestrates a Gemini-powered agent
    to explore Cloud SQL databases and fetch the data.
    """
    print("this is the request payload to nltosql agent: ", request)
    try:
        result = await run_sql_agent(
            query=request.natural_language_query,
            databases=request.databases,
            session_id=request.channel_id
        )
        return result
    except Exception as e:
        print(f"Error in SQL generation endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))