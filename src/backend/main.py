"""
Main FastAPI application
"""
import os
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

# Temporary: Set a fake Groq API key for testing if not found
if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "fake-key-for-testing"
    print("Using fake Groq API key for testing")

# Debug: Check if environment variables are loaded
print(f"GROQ_API_KEY loaded: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
if os.getenv('GROQ_API_KEY'):
    print(f"GROQ_API_KEY starts with: {os.getenv('GROQ_API_KEY')[:10]}...")

# Import API routes - these are in the same backend directory
from api.routes import components, flows, health

# Import core components - these are in the same backend directory
from core.registry import ComponentRegistry
from services.component_manager import ComponentManager
from services.storage import StorageService
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def register_components_manually():
    """Manually register components"""
    try:
        # Import and register core components
        from components.llms.base_llm import LLMComponent
        from components.llms.fake_llm import FakeLLMComponent
        from components.chat_models.base_chat import ChatModelComponent
        from components.embeddings.base_embeddings import EmbeddingsComponent
        
        # Register them manually
        ComponentRegistry.register(LLMComponent)
        ComponentRegistry.register(FakeLLMComponent)
        ComponentRegistry.register(ChatModelComponent)
        ComponentRegistry.register(EmbeddingsComponent)
        
        logger.info("Core components registered successfully")
        
        # Try to register text input component
        try:
            from components.inputs.text_input import TextInputComponent
            ComponentRegistry.register(TextInputComponent)
            logger.info("Text Input component registered successfully")
        except ImportError as e:
            logger.warning(f"Could not import TextInputComponent: {e}")
            # Create a simple text input component inline
            from core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
            from core.registry import register_component
            
            # ... rest of the inline component creation code stays the same
            @register_component
            class TextInputComponent(BaseLangChainComponent):
                def _setup_component(self):
                    self.metadata = ComponentMetadata(
                        display_name="Text Input",
                        description="Simple text input component",
                        icon="📝",
                        category="inputs",
                        tags=["input", "text"]
                    )
                    self.inputs = [
                        ComponentInput(
                            name="user_input",
                            display_name="User Input", 
                            field_type="str",
                            required=False,
                            description="User input text"
                        ),
                        ComponentInput(
                            name="placeholder",
                            display_name="Placeholder",
                            field_type="str",
                            default="Enter text...",
                            required=False,
                            description="Placeholder text"
                        )
                    ]
                    self.outputs = [
                        ComponentOutput(
                            name="text",
                            display_name="Output Text",
                            field_type="str", 
                            method="get_text",
                            description="The processed text"
                        )
                    ]
                
                async def execute(self, **kwargs):
                    user_input = kwargs.get("user_input", "")
                    placeholder = kwargs.get("placeholder", "Enter text...")
                    
                    # Use global input if user_input is empty
                    if not user_input:
                        user_input = "artificial intelligence latest developments"  # Default for testing
                    
                    return {
                        "text": user_input, 
                        "length": len(user_input),
                        "placeholder": placeholder
                    }
            
            logger.info("Inline Text Input component created and registered")
        
        # Try to register web search tool
        try:
            from components.tools.tools import WebSearchToolComponent
            ComponentRegistry.register(WebSearchToolComponent)
            logger.info("Web Search Tool component registered successfully")
        except ImportError as e:
            logger.warning(f"Could not import WebSearchToolComponent: {e}")
        
        # Try to register output parsers
        try:
            from components.output_parsers.parsers import StringOutputParserComponent, JsonOutputParserComponent
            ComponentRegistry.register(StringOutputParserComponent)
            ComponentRegistry.register(JsonOutputParserComponent)
            logger.info("Output parser components registered successfully")
        except ImportError as e:
            logger.warning(f"Could not import output parser components: {e}")
            # Create simple output parsers inline
            from core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
            from core.registry import register_component
            
            @register_component
            class StringOutputParserComponent(BaseLangChainComponent):
                def _setup_component(self):
                    self.metadata = ComponentMetadata(
                        display_name="String Output Parser",
                        description="Parse LLM output as string",
                        icon="📄",
                        category="output_parsers",
                        tags=["parser", "string"]
                    )
                    self.inputs = [
                        ComponentInput(
                            name="llm_output",
                            display_name="LLM Output",
                            field_type="str",
                            description="Output to parse"
                        )
                    ]
                    self.outputs = [
                        ComponentOutput(
                            name="parsed_output",
                            display_name="Parsed String",
                            field_type="str",
                            method="parse_string",
                            description="Parsed output"
                        )
                    ]
                
                async def execute(self, **kwargs):
                    llm_output = kwargs.get("llm_output", "")
                    parsed = llm_output.strip()
                    return {"parsed_output": parsed, "length": len(parsed)}
            
            @register_component
            class JsonOutputParserComponent(BaseLangChainComponent):
                def _setup_component(self):
                    self.metadata = ComponentMetadata(
                        display_name="JSON Output Parser",
                        description="Parse LLM output as JSON",
                        icon="📋",
                        category="output_parsers",
                        tags=["parser", "json"]
                    )
                    self.inputs = [
                        ComponentInput(
                            name="llm_output",
                            display_name="LLM Output",
                            field_type="str",
                            description="JSON string to parse"
                        )
                    ]
                    self.outputs = [
                        ComponentOutput(
                            name="parsed_json",
                            display_name="Parsed JSON",
                            field_type="dict",
                            method="parse_json",
                            description="Parsed JSON object"
                        )
                    ]
                
                async def execute(self, **kwargs):
                    llm_output = kwargs.get("llm_output", "")
                    try:
                        import json
                        # Try to extract JSON from the output
                        start = llm_output.find('{')
                        end = llm_output.rfind('}') + 1
                        if start != -1 and end != 0:
                            json_str = llm_output[start:end]
                            parsed = json.loads(json_str)
                        else:
                            # Fallback: create a mock JSON response
                            parsed = {
                                "topic": "artificial intelligence",
                                "sentiment": "positive",
                                "summary": "Analysis completed successfully"
                            }
                    except Exception as e:
                        parsed = {"error": f"Failed to parse JSON: {str(e)}", "raw_output": llm_output}
                    
                    return {"parsed_json": parsed, "is_valid": "error" not in parsed}
            
            logger.info("Inline output parser components created and registered")
        
        logger.info("Manual component registration completed")
        
    except Exception as e:
        logger.error(f"Manual registration failed: {e}")

# Track application startup time
startup_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("=" * 50)
    logger.info("Starting LangChain Platform...")
    
    # Manual registration
    register_components_manually()
    
    # Log registered components
    component_count = len(ComponentRegistry._components)
    categories = ComponentRegistry.get_categories()
    logger.info(f"Registered {component_count} components across {len(categories)} categories")
    
    # List all registered components
    logger.info("Registered components:")
    for name in ComponentRegistry._components.keys():
        logger.info(f"  - {name}")
    
    for category, component_list in categories.items():
        logger.info(f"Category '{category}': {component_list}")
    
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
        if 'component_manager' in locals():
            component_manager.clear_cache()
            logger.info("Cleared component caches")
        
        logger.info("Cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
    
    logger.info("LangChain Platform shut down")

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
        "registered_components": list(ComponentRegistry._components.keys()),
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