"""
Main FastAPI application
"""
import logging
import time
import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
project_root = backend_dir.parent.parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# Import API routes with absolute imports
from api.routes import components, flows, health
from core.registry import ComponentRegistry
from services.component_manager import ComponentManager
from services.storage import StorageService

# Import all component modules to trigger registration
import components as comp_module

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