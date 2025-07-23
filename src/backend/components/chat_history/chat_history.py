from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import Dict, Any, List, Union

async def execute(self, **kwargs) -> Dict[str, Any]:
    agent_executor = kwargs.get("agent_executor")
    input_query = kwargs.get("input_query")
    chat_history = kwargs.get("chat_history", [])  # Already handling chat history
    return_intermediate_steps = kwargs.get("return_intermediate_steps", True)
    
    # Convert chat history to LangChain messages format
    formatted_chat_history = self._convert_to_langchain_messages(chat_history)
    
    # Handle mock agents for demo
    if agent_executor is None:
        mock_updated_history = formatted_chat_history + [
            HumanMessage(content=input_query),
            AIMessage(content=f"Mock response for query: {input_query}")
        ]
        return {
            "response": f"Mock response for query: {input_query}",
            "intermediate_steps": [],
            "execution_metadata": {
                "total_steps": 0,
                "input_length": len(input_query or ""),
                "output_length": 20,
                "success": True,
                "agent_type": "mock",
                "chat_history_length": len(mock_updated_history)
            },
            "input_query": input_query,
            "updated_chat_history": self._convert_to_serializable(mock_updated_history)
        }
    
    # Prepare input with formatted chat history
    agent_input = {
        "input": input_query,
        "chat_history": formatted_chat_history  # Now properly formatted
    }
    
    # Execute agent
    try:
        result = await agent_executor.ainvoke(
            agent_input,
            return_only_outputs=False,
            include_run_info=True
        )
        
        response = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Format intermediate steps
        formatted_steps = []
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                formatted_steps.append({
                    "action": {
                        "tool": action.tool if hasattr(action, 'tool') else str(action),
                        "tool_input": action.tool_input if hasattr(action, 'tool_input') else "",
                        "log": action.log if hasattr(action, 'log') else ""
                    },
                    "observation": str(observation)
                })
        
        # Create updated chat history with new conversation turn
        updated_chat_history = formatted_chat_history + [
            HumanMessage(content=input_query),
            AIMessage(content=response)
        ]
        
        execution_metadata = {
            "total_steps": len(formatted_steps),
            "input_length": len(input_query),
            "output_length": len(response),
            "success": True,
            "chat_history_length": len(updated_chat_history),
            "previous_history_length": len(formatted_chat_history)
        }
        
    except Exception as e:
        response = f"Agent execution failed: {str(e)}"
        formatted_steps = []
        
        # Still create updated history even on error for continuity
        updated_chat_history = formatted_chat_history + [
            HumanMessage(content=input_query),
            AIMessage(content=response)
        ]
        
        execution_metadata = {
            "error": str(e),
            "success": False,
            "chat_history_length": len(updated_chat_history),
            "previous_history_length": len(formatted_chat_history)
        }
    
    return {
        "response": response,
        "intermediate_steps": formatted_steps,
        "execution_metadata": execution_metadata,
        "input_query": input_query,
        "updated_chat_history": self._convert_to_serializable(updated_chat_history)
    }

def _convert_to_langchain_messages(self, chat_history: List[Union[Dict, str, BaseMessage]]) -> List[BaseMessage]:
    """Convert various chat history formats to LangChain BaseMessage objects"""
    messages = []
    
    for item in chat_history:
        if isinstance(item, BaseMessage):
            # Already a LangChain message
            messages.append(item)
        elif isinstance(item, dict):
            # Convert dict format to message
            role = item.get("role", "human").lower()
            content = item.get("content", "")
            
            if role in ["human", "user"]:
                messages.append(HumanMessage(content=content))
            elif role in ["ai", "assistant", "bot"]:
                messages.append(AIMessage(content=content))
        elif isinstance(item, str):
            # Treat strings as human messages
            messages.append(HumanMessage(content=item))
    
    return messages

def _convert_to_serializable(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Convert LangChain messages back to serializable format"""
    serialized = []
    
    for message in messages:
        if isinstance(message, HumanMessage):
            serialized.append({
                "role": "human",
                "content": message.content,
                "type": "human"
            })
        elif isinstance(message, AIMessage):
            serialized.append({
                "role": "assistant", 
                "content": message.content,
                "type": "ai"
            })
    
    return serialized