"""
Component name mapping for flow validation
"""

# Mapping from frontend component names to backend registered names
COMPONENT_NAME_MAPPING = {
    "Text Input": "Text Input",
    "ChatModel": "Chat Model", 
    "Chat Model": "Chat Model",
    "OpenAI LLM": "OpenAI LLM",
    "LLM Model": "LLM Model",
    "Web Search Tool": "Web Search Tool",
    "OpenAI Functions Agent": "OpenAI Functions Agent",
    "Agent Executor": "Agent Executor",
    "Embeddings": "Embeddings",
    "Vector Store": "Vector Store",
    "Vector Store Retriever": "Vector Store Retriever"
}

def map_component_name(flow_component_name: str) -> str:
    """Map flow component name to registered component name"""
    return COMPONENT_NAME_MAPPING.get(flow_component_name, flow_component_name)