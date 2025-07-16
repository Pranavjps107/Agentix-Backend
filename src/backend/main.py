"""
Main FastAPI application
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# Import API routes
from .api.routes import components, flows, health

# Import core components
from .core.registry import ComponentRegistry
from .services.component_manager import ComponentManager
from .services.storage import StorageService

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
    
    # Import and register all components explicitly
    try:
        logger.info("Importing and registering components...")
        
        # LLM Components
        from .components.llms.base_llm import LLMComponent
        from .components.llms.openai_llm import OpenAILLMComponent
        from .components.llms.anthropic_llm import AnthropicLLMComponent
        from .components.llms.fake_llm import FakeLLMComponent
        
        # Chat Model Components
        from .components.chat_models.base_chat import ChatModelComponent
        
        # Embedding Components
        from .components.embeddings.base_embeddings import EmbeddingsComponent
        
        # Agent Components
        from .components.agents.agents import (
            OpenAIFunctionsAgentComponent, 
            ReActAgentComponent, 
            AgentExecutorComponent
        )
        
        # Tool Components
        from .components.tools.tools import (
            CustomToolComponent, 
            PythonREPLToolComponent, 
            WebSearchToolComponent
        )
        
        # Document Loader Components
        from .components.document_loaders.loaders import (
            TextLoaderComponent, 
            PDFLoaderComponent, 
            WebLoaderComponent, 
            CSVLoaderComponent
        )
        
        # Output Parser Components
        from .components.output_parsers.parsers import (
            StringOutputParserComponent, 
            JsonOutputParserComponent, 
            ListOutputParserComponent
        )
        
        # Prompt Components
        from .components.prompts.prompt_templates import (
            PromptTemplateComponent, 
            ChatPromptTemplateComponent
        )
        
        # Vector Store Components
        from .components.vectorstores.vectorstore import (
            VectorStoreComponent, 
            VectorStoreRetrieverComponent
        )
        
        logger.info("All component modules imported successfully")
        
        # Verify components are registered (they should auto-register via @register_component decorator)
        component_count = len(ComponentRegistry._components)
        if component_count == 0:
            logger.warning("No components auto-registered, attempting manual registration...")
            
            # Manual registration fallback
            component_classes = [
                LLMComponent,
                OpenAILLMComponent, 
                AnthropicLLMComponent,
                FakeLLMComponent,
                ChatModelComponent,
                EmbeddingsComponent,
                OpenAIFunctionsAgentComponent,
                ReActAgentComponent,
                AgentExecutorComponent,
                CustomToolComponent,
                PythonREPLToolComponent,
                WebSearchToolComponent,
                TextLoaderComponent,
                PDFLoaderComponent,
                WebLoaderComponent,
                CSVLoaderComponent,
                StringOutputParserComponent,
                JsonOutputParserComponent,
                ListOutputParserComponent,
                PromptTemplateComponent,
                ChatPromptTemplateComponent,
                VectorStoreComponent,
                VectorStoreRetrieverComponent
            ]
            
            for component_class in component_classes:
                try:
                    ComponentRegistry.register(component_class)
                    logger.info(f"Manually registered: {component_class.__name__}")
                except Exception as e:
                    logger.error(f"Failed to register {component_class.__name__}: {e}")
        
    except ImportError as e:
        logger.error(f"Failed to import some components: {e}")
        logger.info("Continuing with available components...")
    except Exception as e:
        logger.error(f"Error during component registration: {e}")
        raise
    
    # Log final component registration status
    component_count = len(ComponentRegistry._components)
    categories = ComponentRegistry.get_categories()
    logger.info(f"Successfully registered {component_count} components across {len(categories)} categories")
    
    if component_count > 0:
        for category, component_list in categories.items():
            logger.info(f"  📁 {category}: {len(component_list)} components")
            for comp_name in component_list:
                logger.info(f"    ✓ {comp_name}")
    else:
        logger.warning("⚠️  No components registered! Check component imports and decorators.")
    
    # Initialize services
    try:
        storage_service = StorageService()
        logger.info("✓ Storage service initialized")
        
        component_manager = ComponentManager()
        logger.info("✓ Component manager initialized")
        
        logger.info("🚀 LangChain Platform started successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LangChain Platform...")
    
    # Cleanup resources
    try:
        # Clear caches
        component_manager.clear_cache()
        logger.info("✓ Cleared component caches")
        
        # Any other cleanup
        logger.info("✓ Cleanup completed")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {str(e)}")
    
    logger.info("👋 LangChain Platform shut down")

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
    - Chat Models (Conversational AI)
    - Embeddings (OpenAI, HuggingFace, etc.)
    - Vector Stores (Chroma, Pinecone, etc.)
    - Tools and Agents
    - Document Loaders
    - Output Parsers
    - Prompt Templates
    - And much more!
    
    ## API Endpoints
    - `/api/v1/components/` - Component management
    - `/api/v1/flows/` - Flow execution and management
    - `/api/v1/health/` - Health checks and monitoring
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
    
    # Only log non-health check requests to reduce noise
    if not request.url.path.startswith("/api/v1/health"):
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
            "path": str(request.url.path),
            "timestamp": time.time()
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
        "uptime_formatted": format_uptime(uptime),
        "components_registered": len(ComponentRegistry._components),
        "categories": list(ComponentRegistry.get_categories().keys()),
        "total_categories": len(ComponentRegistry.get_categories()),
        "endpoints": {
            "documentation": "/docs",
            "redoc": "/redoc", 
            "health": "/api/v1/health",
            "detailed_health": "/api/v1/health/detailed",
            "components": "/api/v1/components",
            "flows": "/api/v1/flows",
            "flow_templates": "/api/v1/flows/templates"
        },
        "features": [
            "Drag-and-drop flow builder",
            "Real-time component execution", 
            "LangChain integration",
            "Flow export to Python",
            "Built-in monitoring",
            "Caching and optimization",
            "Multi-provider LLM support",
            "Vector store integration",
            "Agent framework",
            "Tool ecosystem"
        ]
    }

@app.get("/info")
async def get_platform_info():
    """Get detailed platform information"""
    component_stats = ComponentRegistry.get_stats()
    categories = ComponentRegistry.get_categories()
    
    return {
        "platform": {
            "name": "LangChain Platform",
            "version": "1.0.0",
            "uptime_seconds": time.time() - startup_time,
            "uptime_formatted": format_uptime(time.time() - startup_time)
        },
        "components": {
            "total_registered": component_stats.get("total_components", 0),
            "categories_count": component_stats.get("categories", 0),
            "by_category": component_stats.get("components_by_category", {}),
            "detailed_categories": categories
        },
        "system": {
            "python_version": "3.11+",
            "framework": "FastAPI",
            "database": "File-based storage",
            "caching": "Redis/In-memory",
            "monitoring": "Prometheus metrics"
        },
        "capabilities": {
            "llm_providers": ["OpenAI", "Anthropic", "HuggingFace", "Local models"],
            "vector_stores": ["Chroma", "Pinecone", "Qdrant", "Weaviate", "FAISS"],
            "document_loaders": ["PDF", "Text", "Web", "CSV", "JSON"],
            "agents": ["OpenAI Functions", "ReAct", "Custom tools"],
            "output_formats": ["String", "JSON", "Structured", "Lists"]
        }
    }

@app.get("/status")
async def get_status():
    """Quick status check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - startup_time,
        "uptime_formatted": format_uptime(time.time() - startup_time),
        "components": len(ComponentRegistry._components),
        "categories": len(ComponentRegistry.get_categories())
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
                "uptime_formatted": format_uptime(time.time() - startup_time),
                "registered_components": len(ComponentRegistry._components),
                "categories": len(ComponentRegistry.get_categories())
            },
            "execution_metrics": stats,
            "component_stats": ComponentRegistry.get_stats(),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# Serve static files (for frontend)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✓ Static files mounted at /static")
except Exception as e:
    logger.warning(f"⚠️  Static files directory not found: {str(e)}")

def format_uptime(uptime_seconds: float) -> str:
    """Format uptime in human readable format"""
    days = int(uptime_seconds // 86400)
    hours = int((uptime_seconds % 86400) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

# Development server configuration
if __name__ == "__main__":
    uvicorn.run(
        "src.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        workers=1  # Use single worker for development
    )