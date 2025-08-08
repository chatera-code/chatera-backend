# In services/sql_agent/agent.py

import logging
import httpx
import json
import os
from core.config import engine_mysql
from core.config import generation_model
from typing import Any
from services.sql_agent.tools import TOOL_REGISTRY
from services.sql_agent.gemini_config import get_system_prompt, GEMINI_TOOLS

session_dict = {}

MAIN_APP_NOTIFY_URL = os.getenv("MAIN_APP_NOTIFY_URL", "http://main_app:8000/internal/notify")

async def send_update_event(client: httpx.AsyncClient, channel_id: str, event_type: str, message: str):
    """Helper function to send a real-time event back to the main app."""
    try:
        payload = {
            "channel_id": channel_id,
            "event_data": {"type": event_type, "message": message}
        }
        await client.post(MAIN_APP_NOTIFY_URL, json=payload)
    except Exception as e:
        logging.error(f"Failed to send update event for channel {channel_id}: {e}")
        
async def run_sql_agent(query: str, databases: list[str], session_id: str) -> dict:
    """The main agentic loop for the Text-to-SQL agent."""

    if engine_mysql is None:
        return {
            "status": "error", "final_query": None,
            "final_response": "Database connection is not configured or failed to initialize.",
            "history": []
        }
        
    if session_id not in session_dict:
        session_dict[session_id] = {
            "conv_history": [
                {'role': 'user', 'parts': [{'text': get_system_prompt(databases)}]},
                {'role': 'model', 'parts': [{'text': "Understood. I am ready to assist."}]},
            ]}
    
    history = session_dict[session_id]["conv_history"]
    history.append({'role': 'user', 'parts': [{'text': query}]})
    max_loops = 15
    loop_count = 0

    async with httpx.AsyncClient() as client:
        while True:
            if loop_count >= max_loops:
                logging.error(f"Agent loop exceeded max iterations of {max_loops}.")
                return { "status": "error", "final_response": "Request timed out.", "history": history }
            
            logging.info(f"\n--- SQL Agent Loop {loop_count+1} ---")
            loop_count += 1
            response = generation_model.generate_content(history, tools=GEMINI_TOOLS)
            model_response_part = response.candidates[0].content
            
            if model_response_part.parts[0].function_call.name:
                history.append({'role': 'model', 'parts': model_response_part.parts})
                tool_responses_for_this_turn = []
                
                for part in model_response_part.parts:
                    tool_call = part.function_call
                    tool_name = tool_call.name
                    tool_params = {key: value for key, value in tool_call.args.items()}

                    user_friendly_message = f"I need to use the `{tool_name}` tool." # Default message
                    if tool_name == 'fetch_tables':
                        user_friendly_message = f"🔍 Exploring the `{tool_params.get('db_name')}` database to see what tables are available..."
                    elif tool_name == 'fetch_columns':
                        user_friendly_message = f"📄 Examining the columns in the `{tool_params.get('table_name')}` table..."
                    elif tool_name == 'validate_sql':
                        user_friendly_message = "✅ Checking if my generated query is valid..."
                    elif tool_name == 'execute_sql_and_fetch_results':
                        user_friendly_message = "🚀 Running the final query to get your answer..."

                    await send_update_event(client, session_id, "status", user_friendly_message)
                        
                    print(f"Gemini wants to call tool: {tool_name} with params: {tool_params}")
                    
                    if tool_name in TOOL_REGISTRY:
                        tool_function = TOOL_REGISTRY[tool_name]
                        
                        # Execute the tool
                        tool_result = tool_function(engine=engine_mysql, **tool_params)
                        
                        if isinstance(tool_result, dict) and "status" in tool_result and tool_result["status"] == "error":
                            tool_result = tool_result["message"]
                            print(f"tool returned error: ", tool_result)
                        
                        # Append the result to our list for this turn
                        tool_responses_for_this_turn.append({
                            "function_response": {
                                "name": tool_name,
                                "response": {"content": tool_result}
                            }
                        })
                    else:
                        # Handle unknown tool
                        tool_responses_for_this_turn.append({
                            "function_response": {
                                "name": tool_name,
                                "response": {"content": "Error: Unknown tool."}
                            }
                        })

                # Add all the tool results to history in a single turn
                history.append({'role': 'function', 'parts': tool_responses_for_this_turn})
                
            else:
                # Agent provides final text answer
                final_text = model_response_part.parts[0].text
                history.append({'role': 'model', 'parts': model_response_part.parts})
                try:
                    res = json.loads(final_text)
                    final_text = res["error_message"]
                    print("gemini returned error: ", final_text)
                    return {
                        "status": "error",
                        "final_response": final_text, 
                    }
                    
                except:
                    return {
                        "status": "success",
                        "final_response": final_text, 
                    }
