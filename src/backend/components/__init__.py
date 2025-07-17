"""
LangChain Components Package
Contains all component implementations organized by category
"""

# Import all component modules to ensure registration
try:
    from . import agents
    from . import callbacks  
    from . import chat_models
    from . import embeddings
    from . import llms
    from . import memory
    from . import output_parsers
    from . import prompts
    from . import retrievers
    from . import runnables
    from . import tools
    from . import vectorstores
    from . import document_loaders
    
    # Import specific components to ensure they're registered
    from .llms.base_llm import LLMComponent
    from .llms.openai_llm import OpenAILLMComponent
    from .llms.anthropic_llm import AnthropicLLMComponent
    from .llms.fake_llm import FakeLLMComponent
    
    from .chat_models.base_chat import ChatModelComponent
    from .embeddings.base_embeddings import EmbeddingsComponent
    from .agents.agents import OpenAIFunctionsAgentComponent, ReActAgentComponent, AgentExecutorComponent
    from .tools.tools import WebSearchToolComponent, CustomToolComponent
    from .vectorstores.vectorstore import VectorStoreComponent, VectorStoreRetrieverComponent
    from .document_loaders.loaders import TextLoaderComponent, PDFLoaderComponent, WebLoaderComponent
    from .output_parsers.parsers import StringOutputParserComponent, JsonOutputParserComponent
    from .prompts.prompt_templates import PromptTemplateComponent, ChatPromptTemplateComponent
    
except ImportError as e:
    import logging
    logging.warning(f"Failed to import some component modules: {e}")

__all__ = [
    "agents",
    "callbacks", 
    "chat_models",
    "embeddings",
    "llms",
    "memory",
    "output_parsers",
    "prompts",
    "retrievers",
    "runnables",
    "tools",
    "vectorstores",
    "document_loaders"
]