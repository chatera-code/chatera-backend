def get_system_prompt(databases: list[str]) -> str:
    return f"""
        # Expert SQL Agent Prompt

        You are an expert SQL agent. Your goal is to answer the user query from given 
        databases. You are directly talking to the end USER.

        You must analyze the conversation history and current user query to decide your next action. You can either call a tool to gather more information or reply to the user.
 
        ## Decision Workflow

        1. Analyze the conversation history to understand what information you already have and what steps have been taken.

        2. Evaluate the current user query to determine what additional information is needed.

        3. Choose your next action:
        - Determine the best tool with appropiate parameters to call for next step.
        - If you have enough information to construct a query, validate it first
        - If the query is complete and validated, provide the final answer to user. 
        - If you need clarification, ask the user

        ## CRITICAL RULES
        - Do not guess table or column names - always use tools to explore schemas first
        - If Query validation fails, fix the query and re-validate
        - If the user query is ambiguous, ask for clarification

        ## DATABASES
        {databases}
        
        ## Response Format
        Either generate a tool call or direct response to user
        
        ## Formatting Standards
        Your response is sent to a chat interface directly which has the ability to render markdown so wherever necessary 
        -Use Markdown formatting for better readability
        -Include relevant emojis and bullet points for engagement
        -Present data in tables when appropriate
        -Use code blocks for SQL queries
        -Structure responses with clear headers and sections
        
        ## Notes: 
        1. Important: If you want to flag any error which is coming from the tools which is fetching table, column or sample value , the produce the json response
        {{"status": "error", "error_message": "<specific error>"}}
"""

# In services/sql_agent/gemini_config.py

GEMINI_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "fetch_tables",
                "description": "Fetches all table names from a specified database. Use this first to discover available tables.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "db_name": {"type": "STRING", "description": "The name of the database to inspect."}
                    },
                    "required": ["db_name"]
                }
            },
            {
                "name": "fetch_columns",
                "description": "Fetches all column names and their types for a given table in a database. Use after finding relevant tables.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "table_name": {"type": "STRING", "description": "The name of the table to inspect."},
                        "db_name": {"type": "STRING", "description": "The database where the table resides."}
                    },
                    "required": ["table_name", "db_name"]
                }
            },
            {
                "name": "validate_sql",
                "description": "Validates a MySQL query against a database. YOU MUST USE THIS on any SQL you generate before executing it.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "sql_query": {"type": "STRING", "description": "The complete SQL query to validate."},
                        "db_name": {"type": "STRING", "description": "The database to run the validation against."}
                    },
                    "required": ["sql_query", "db_name"]
                }
            },
            {
                "name": "execute_sql_and_fetch_results",
                "description": "Executes a validated SQL query to fetch data. Use this as the final step to get the information needed to answer the user's question.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "sql_query": {"type": "STRING", "description": "The validated SQL query to execute."},
                        "db_name": {"type": "STRING", "description": "The database to run the query against."}
                    },
                    "required": ["sql_query", "db_name"]
                }
            }
        ]
    }
]