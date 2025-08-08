# In services/sql_agent/tools.py

import logging
import json
import sqlalchemy
import traceback
from sqlalchemy.exc import SQLAlchemyError

def fetch_tables(engine: sqlalchemy.engine.Engine, db_name: str) -> str:
    """Fetches all table names from the specified database."""
    logging.info(f"TOOL CALLED: fetch_tables(db_name='{db_name}')")
    try:
        with engine.connect() as connection:
            inspector = sqlalchemy.inspect(connection)
            
            # --- CORRECTED ---
            # Explicitly pass the db_name as the 'schema' argument.
            # This is the robust SQLAlchemy pattern for reflection.
            table_names = inspector.get_table_names(schema=db_name)
            
            return str(table_names)
    except Exception as e:
        print(traceback.format_exc())
        return {"status": "error", "message": f"Error fetching tables from '{db_name}': {e}"}

def fetch_columns(engine: sqlalchemy.engine.Engine, db_name: str, table_name: str) -> str:
    """Fetches all column names and their types for a given table in a specific database."""
    logging.info(f"TOOL CALLED: fetch_columns(db_name='{db_name}', table_name='{table_name}')")
    try:
        with engine.connect() as connection:
            inspector = sqlalchemy.inspect(connection)

            # --- CORRECTED ---
            # Apply the same explicit pattern to get_columns.
            columns = inspector.get_columns(table_name, schema=db_name)
            
            return str([f"{col['name']} ({col['type']})" for col in columns])
    except Exception as e:
        return {"status": "error", "message": f"Error fetching columns for table '{table_name}' in '{db_name}': {e}"}
         

def validate_sql(engine: sqlalchemy.engine.Engine, db_name: str, sql_query: str) -> str:
    """Validates a SQL query by executing it against the specified database."""
    logging.info(f"TOOL CALLED: validate_sql(db_name='{db_name}')")
    try:
        # The `with engine.connect()` block handles the transaction lifecycle.
        with engine.connect() as connection:
            # Set the database context for this specific connection
            connection.execute(sqlalchemy.text(f"USE `{db_name}`"))
            
            if "limit" not in sql_query.lower():
                 sql_query += " LIMIT 1"
            connection.execute(sqlalchemy.text(sql_query))

            
        return "Validation successful. The SQL query is valid."
    except SQLAlchemyError as e:
        return f"Validation failed for database '{db_name}'. Error: {e}. You MUST fix the query."

def execute_sql_and_fetch_results(engine: sqlalchemy.engine.Engine, db_name: str, sql_query: str) -> str:
    """Executes a SQL query against a specific database and returns the results."""
    logging.info(f"TOOL CALLED: execute_sql_and_fetch_results(db_name='{db_name}')")
    if "limit" not in sql_query.lower():
        sql_query = f"{sql_query.rstrip(';')} LIMIT 25;"

    try:
        with engine.connect() as connection:
            # --- NO CHANGE NEEDED ---
            # Like validate_sql, this function executes raw SQL,
            # so `USE` is the correct approach here.
            connection.execute(sqlalchemy.text(f"USE `{db_name}`"))
            
            result = connection.execute(sqlalchemy.text(sql_query))
            column_names = result.keys()
            results_as_dicts = [dict(zip(column_names, row)) for row in result.fetchall()]
            
            if not results_as_dicts:
                return "Query executed successfully, but it returned no results."
            return json.dumps(results_as_dicts)
    except SQLAlchemyError as e:
        return {"status": "error", "message": f"Execution failed in database '{db_name}'. Error: {e}"}
         

# The tool registry mapping remains the same.
TOOL_REGISTRY = {
    "fetch_tables": fetch_tables,
    "fetch_columns": fetch_columns,
    "validate_sql": validate_sql,
    "execute_sql_and_fetch_results": execute_sql_and_fetch_results,
}