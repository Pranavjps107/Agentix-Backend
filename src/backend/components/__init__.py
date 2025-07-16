# src/backend/components/__init__.py
"""
LangChain Components Package
Contains all component implementations organized by category
"""

# Import all component modules to ensure registration
try:
    # Import all submodules
    from . import llms
    from . import chat_models
    from . import embeddings
    from . import agents
    from . import tools
    from . import document_loaders
    from . import output_parsers
    from . import prompts
    from . import vectorstores
    
    # Import specific components to trigger registration
    from .llms.base_llm import LLMComponent
    from .llms.openai_llm import OpenAILLMComponent
    from .llms.anthropic_llm import AnthropicLLMComponent
    from .llms.fake_llm import FakeLLMComponent
    
    from .chat_models.base_chat import ChatModelComponent
    from .embeddings.base_embeddings import EmbeddingsComponent
    
    from .agents.agents import OpenAIFunctionsAgentComponent, ReActAgentComponent, AgentExecutorComponent
    from .tools.tools import CustomToolComponent, PythonREPLToolComponent, WebSearchToolComponent
    from .document_loaders.loaders import TextLoaderComponent, PDFLoaderComponent, WebLoaderComponent, CSVLoaderComponent
    from .output_parsers.parsers import StringOutputParserComponent, JsonOutputParserComponent, ListOutputParserComponent
    from .prompts.prompt_templates import PromptTemplateComponent, ChatPromptTemplateComponent
    from .vectorstores.vectorstore import VectorStoreComponent, VectorStoreRetrieverComponent
    
except ImportError as e:
    import logging
    logging.warning(f"Failed to import some component modules: {e}")

__all__ = [
    "llms",
    "chat_models", 
    "embeddings",
    "agents",
    "tools",
    "document_loaders",
    "output_parsers",
    "prompts",
    "vectorstores"
]