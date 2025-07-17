"""
Main FastAPI application
"""
import logging
import time
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import API routes - using absolute imports
from src.backend.api.routes import components, flows, health

# Import core components to ensure registration
from src.backend.core.registry import ComponentRegistry
from src.backend.services.component_manager import ComponentManager
from src.backend.services.storage import StorageService

# CRITICAL: Import ALL component modules to trigger registration
# This ensures components are registered before the server starts

# Add the components path to ensure imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

# Import all component categories - this triggers @register_component decorators
try:
    from src.backend.components.llms import *
    from src.backend.components.chat_models import *
    from src.backend.components.embeddings import *
    from src.backend.components.agents import *
    from src.backend.components.tools import *
    from src.backend.components.document_loaders import *
    from src.backend.components.vectorstores import *
    from src.backend.components.output_parsers import *
    from src.backend.components.prompts import *
    from src.backend.components.retrievers import *
    from src.backend.components.memory import *
    from src.backend.components.callbacks import *
    from src.backend.components.runnables import *
    
    # Import any remaining components
    from src.backend.components import *
    
except ImportError as e:
    logging.warning(f"Failed to import some component modules: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track application startup time
startup_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("=" * 50)
    logger.info("Starting LangChain Platform...")
    
    # Force registration of all components
    _register_missing_components()
    
    # Log registered components
    component_count = len(ComponentRegistry._components)
    categories = ComponentRegistry.get_categories()
    logger.info(f"Registered {component_count} components across {len(categories)} categories")
    
    for category, component_list in categories.items():
        logger.info(f"  {category}: {len(component_list)} components")
    
    # Initialize services
    try:
        storage_service = StorageService()
        logger.info("Storage service initialized")
        
        component_manager = ComponentManager()
        logger.info("Component manager initialized")
        
        logger.info("LangChain Platform started successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LangChain Platform...")
    
    # Cleanup resources
    try:
        # Clear caches
        component_manager.clear_cache()
        logger.info("Cleared component caches")
        
        # Any other cleanup
        logger.info("Cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
    
    logger.info("LangChain Platform shut down")

def _register_missing_components():
    """Register commonly needed components that might be missing"""
    from src.backend.core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
    from src.backend.core.registry import register_component
    
    # Register Text Input component
    @register_component
    class TextInputComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="Text Input",
                description="Text input component for user queries",
                icon="📝",
                category="input",
                tags=["input", "text"]
            )
            self.inputs = [
                ComponentInput(
                    name="placeholder",
                    display_name="Placeholder",
                    field_type="str",
                    default="Enter text...",
                    required=False,
                    description="Placeholder text"
                ),
                ComponentInput(
                    name="text",
                    display_name="Text",
                    field_type="str",
                    default="",
                    required=False,
                    description="Input text"
                )
            ]
            self.outputs = [
                ComponentOutput(
                    name="text",
                    display_name="Text Output",
                    field_type="str",
                    method="get_text",
                    description="Input text"
                )
            ]
        
        async def execute(self, **kwargs):
            text = kwargs.get("text", "")
            return {"text": text}
    
    # Register Chat Model component
    @register_component
    class ChatModelComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="Chat Model",
                description="Chat-based language model for conversations",
                icon="💬",
                category="language_models",
                tags=["chat", "conversation", "messages", "ai"]
            )
            self.inputs = [
                ComponentInput(
                    name="provider",
                    display_name="Provider",
                    field_type="str",
                    options=["openai", "anthropic", "google", "fake"],
                    default="fake",
                    description="Chat model provider"
                ),
                ComponentInput(
                    name="model",
                    display_name="Model",
                    field_type="str",
                    default="gpt-3.5-turbo",
                    description="Chat model name"
                ),
                ComponentInput(
                    name="messages",
                    display_name="Messages",
                    field_type="list",
                    default=[],
                    required=False,
                    description="List of chat messages"
                ),
                ComponentInput(
                    name="temperature",
                    display_name="Temperature",
                    field_type="float",
                    default=0.7,
                    required=False,
                    description="Controls randomness in responses"
                ),
                ComponentInput(
                    name="max_tokens",
                    display_name="Max Tokens",
                    field_type="int",
                    default=512,
                    required=False,
                    description="Maximum tokens in response"
                )
            ]
            self.outputs = [
                ComponentOutput(
                    name="response",
                    display_name="Chat Response",
                    field_type="str",
                    method="generate_response",
                    description="The chat model's response"
                ),
                ComponentOutput(
                    name="message_object",
                    display_name="Message Object",
                    field_type="dict",
                    method="get_message_object",
                    description="Full message object"
                )
            ]
        
        async def execute(self, **kwargs):
            provider = kwargs.get("provider", "fake")
            model = kwargs.get("model", "gpt-3.5-turbo")
            messages = kwargs.get("messages", [])
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 512)
            
            # For fake provider, return mock response
            if provider == "fake":
                response = f"Mock response from {model}"
                message_object = {
                    "content": response,
                    "role": "assistant",
                    "provider": provider,
                    "model": model
                }
                return {
                    "response": response,
                    "message_object": message_object
                }
            
            # For real providers, you would implement actual chat model logic here
            return {
                "response": f"Response from {provider} {model}",
                "message_object": {
                    "content": f"Response from {provider} {model}",
                    "role": "assistant",
                    "provider": provider,
                    "model": model
                }
            }
    
    # Register OpenAI Functions Agent component
    @register_component
    class OpenAIFunctionsAgentComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="OpenAI Functions Agent",
                description="Agent that uses OpenAI function calling",
                icon="🤖",
                category="agents",
                tags=["agent", "openai", "functions"]
            )
            self.inputs = [
                ComponentInput(
                    name="llm",
                    display_name="Language Model",
                    field_type="dict",
                    description="Chat model for the agent"
                ),
                ComponentInput(
                    name="tools",
                    display_name="Tools",
                    field_type="list",
                    default=[],
                    description="List of tools available to the agent"
                ),
                ComponentInput(
                    name="system_message",
                    display_name="System Message",
                    field_type="str",
                    default="You are a helpful assistant.",
                    required=False,
                    description="System prompt for the agent"
                ),
                ComponentInput(
                    name="max_iterations",
                    display_name="Max Iterations",
                    field_type="int",
                    default=10,
                    required=False,
                    description="Maximum number of iterations"
                )
            ]
            self.outputs = [
                ComponentOutput(
                    name="agent",
                    display_name="Agent",
                    field_type="dict",
                    method="create_agent",
                    description="Configured agent"
                )
            ]
        
        async def execute(self, **kwargs):
            llm = kwargs.get("llm", {})
            tools = kwargs.get("tools", [])
            system_message = kwargs.get("system_message", "You are a helpful assistant.")
            max_iterations = kwargs.get("max_iterations", 10)
            
            agent_config = {
                "type": "openai_functions",
                "llm": llm,
                "tools": tools,
                "system_message": system_message,
                "max_iterations": max_iterations
            }
            
            return {"agent": agent_config}
    
    # Register Agent Executor component
    @register_component
    class AgentExecutorComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="Agent Executor",
                description="Execute agent with input query",
                icon="⚡",
                category="agents",
                tags=["agent", "executor"]
            )
            self.inputs = [
                ComponentInput(
                    name="agent",
                    display_name="Agent",
                    field_type="dict",
                    description="Agent to execute"
                ),
                ComponentInput(
                    name="input_query",
                    display_name="Input Query",
                    field_type="str",
                    description="Query to send to the agent"
                ),
                ComponentInput(
                    name="return_intermediate_steps",
                    display_name="Return Intermediate Steps",
                    field_type="bool",
                    default=True,
                    required=False,
                    description="Return reasoning steps"
                )
            ]
            self.outputs = [
                ComponentOutput(
                    name="response",
                    display_name="Response",
                    field_type="str",
                    method="execute_agent",
                    description="Agent response"
                ),
                ComponentOutput(
                    name="intermediate_steps",
                    display_name="Intermediate Steps",
                    field_type="list",
                    method="get_steps",
                    description="Reasoning steps"
                )
            ]
        
        async def execute(self, **kwargs):
            agent = kwargs.get("agent", {})
            input_query = kwargs.get("input_query", "")
            return_intermediate_steps = kwargs.get("return_intermediate_steps", True)
            
            # Mock agent execution
            response = f"Agent processed: {input_query}"
            
            intermediate_steps = []
            if return_intermediate_steps:
                intermediate_steps = [
                    {
                        "step": 1,
                        "action": "thinking",
                        "result": f"Processing query: {input_query}"
                    },
                    {
                        "step": 2,
                        "action": "response",
                        "result": response
                    }
                ]
            
            return {
                "response": response,
                "intermediate_steps": intermediate_steps
            }
    
    # Register Web Search Tool
    @register_component
    class WebSearchToolComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="Web Search Tool",
                description="Search the web for information",
                icon="🔍",
                category="tools",
                tags=["search", "web", "information"]
            )
            self.inputs = [
                ComponentInput(
                    name="search_provider",
                    display_name="Search Provider",
                    field_type="str",
                    options=["ddg", "serper", "tavily"],
                    default="ddg",
                    description="Web search provider"
                ),
                ComponentInput(
                    name="num_results",
                    display_name="Number of Results",
                    field_type="int",
                    default=5,
                    required=False,
                    description="Number of search results"
                )
            ]
            self.outputs = [
                ComponentOutput(
                    name="tool",
                    display_name="Search Tool",
                    field_type="dict",
                    method="create_search_tool",
                    description="Web search tool"
                )
            ]
        
        async def execute(self, **kwargs):
            provider = kwargs.get("search_provider", "ddg")
            num_results = kwargs.get("num_results", 5)
            
            tool_config = {
                "type": "web_search",
                "provider": provider,
                "num_results": num_results
            }
            
            return {"tool": tool_config}
# Create FastAPI app
app = FastAPI(
    title="LangChain Drag-and-Drop Platform",
    description="""
    Production-ready LangChain component platform with drag-and-drop interface.
    
    ## Features
    - 🔗 Complete LangChain Core API integration
    - 🎨 Drag-and-drop visual flow builder
    - ⚡ Real-time component execution
    - 📊 Built-in monitoring and caching
    - 🔄 Flow export to Python code
    - 🛡️ Production-ready architecture
    
    ## Components
    - Language Models (OpenAI, Anthropic, etc.)
    - Embeddings (OpenAI, HuggingFace, etc.)
    - Vector Stores (Chroma, Pinecone, etc.)
    - Tools and Agents
    - Document Loaders
    - Output Parsers
    - And much more!
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url.path)
        }
    )

# Include API routers
app.include_router(components.router)
app.include_router(flows.router)
app.include_router(health.router)

# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with platform information"""
    uptime = time.time() - startup_time
    
    return {
        "message": "🚀 LangChain Drag-and-Drop Platform",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": uptime,
        "components_registered": len(ComponentRegistry._components),
        "categories": list(ComponentRegistry.get_categories().keys()),
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc", 
            "health": "/api/v1/health",
            "components": "/api/v1/components",
            "flows": "/api/v1/flows"
        },
        "features": [
            "Drag-and-drop flow builder",
            "Real-time component execution", 
            "LangChain integration",
            "Flow export to Python",
            "Built-in monitoring",
            "Caching and optimization"
        ]
    }

@app.get("/info")
async def get_platform_info():
    """Get detailed platform information"""
    component_stats = ComponentRegistry.get_stats()
    
    return {
        "platform": {
            "name": "LangChain Platform",
            "version": "1.0.0",
            "uptime_seconds": time.time() - startup_time
        },
        "components": component_stats,
        "categories": ComponentRegistry.get_categories(),
        "system": {
            "python_version": "3.11+",
            "framework": "FastAPI",
            "database": "File-based storage"
        }
    }

# Serve static files (for frontend)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Static files mounted at /static")
except Exception as e:
   logger.warning(f"Static files directory not found: {str(e)}")

# Additional utility endpoints
@app.get("/status")
async def get_status():
   """Quick status check"""
   return {
       "status": "healthy",
       "timestamp": time.time(),
       "uptime": time.time() - startup_time,
       "components": len(ComponentRegistry._components)
   }

@app.get("/metrics")
async def get_metrics():
   """Basic metrics endpoint"""
   try:
       component_manager = ComponentManager()
       stats = component_manager.get_manager_stats()
       
       return {
           "platform_metrics": {
               "uptime_seconds": time.time() - startup_time,
               "registered_components": len(ComponentRegistry._components),
               "categories": len(ComponentRegistry.get_categories())
           },
           "execution_metrics": stats,
           "timestamp": time.time()
       }
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

if __name__ == "__main__":
   uvicorn.run(
       "main:app",
       host="0.0.0.0",
       port=8000,
       reload=True,
       log_level="info",
       access_log=True
   )